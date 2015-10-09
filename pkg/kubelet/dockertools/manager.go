/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dockertools

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/golang/groupcache/lru"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/hairpin"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	maxReasonCacheEntries = 200

	// ndots specifies the minimum number of dots that a domain name must contain for the resolver to consider it as FQDN (fully-qualified)
	// we want to able to consider SRV lookup names like _dns._udp.kube-dns.default.svc to be considered relative.
	// hence, setting ndots to be 5.
	ndotsDNSOption = "options ndots:5\n"
	// In order to avoid unnecessary SIGKILLs, give every container a minimum grace
	// period after SIGTERM. Docker will guarantee the termination, but SIGTERM is
	// potentially dangerous.
	// TODO: evaluate whether there are scenarios in which SIGKILL is preferable to
	// SIGTERM for certain process types, which may justify setting this to 0.
	minimumGracePeriodInSeconds = 2

	kubernetesNameLabel                   = "io.kubernetes.pod.name"
	kubernetesPodLabel                    = "io.kubernetes.pod.data"
	kubernetesTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"
	kubernetesContainerLabel              = "io.kubernetes.container.name"

	DockerNetnsFmt = "/proc/%v/ns/net"
)

// DockerManager implements the Runtime interface.
var _ kubecontainer.Runtime = &DockerManager{}

// TODO: make this a TTL based pull (if image older than X policy, pull)
var podInfraContainerImagePullPolicy = api.PullIfNotPresent

type DockerManager struct {
	client              DockerInterface
	recorder            record.EventRecorder
	containerRefManager *kubecontainer.RefManager
	os                  kubecontainer.OSInterface
	machineInfo         *cadvisorApi.MachineInfo

	// The image name of the pod infra container.
	podInfraContainerImage string
	// reasonCache stores the failure reason of the last container creation
	// and/or start in a string, keyed by <pod_UID>_<container_name>. The goal
	// is to propagate this reason to the container status. This endeavor is
	// "best-effort" for two reasons:
	//   1. The cache is not persisted.
	//   2. We use an LRU cache to avoid extra garbage collection work. This
	//      means that some entries may be recycled before a pod has been
	//      deleted.
	reasonCache reasonInfoCache
	// TODO(yifan): Record the pull failure so we can eliminate the image checking
	// in GetPodStatus()?
	// Lower level docker image puller.
	dockerPuller DockerPuller

	// wrapped image puller.
	imagePuller kubecontainer.ImagePuller

	// Root of the Docker runtime.
	dockerRoot string

	// Directory of container logs.
	containerLogsDir string

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// Health check prober.
	prober prober.Prober

	// Generator of runtime container options.
	generator kubecontainer.RunContainerOptionsGenerator

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner

	// Handler used to execute commands in containers.
	execHandler ExecHandler

	// Used to set OOM scores of processes.
	oomAdjuster *oom.OOMAdjuster

	// Get information from /proc mount.
	procFs procfs.ProcFsInterface

	// If true, enforce container cpu limits with CFS quota support
	cpuCFSQuota bool

	// Container GC manager
	containerGC *containerGC
}

func NewDockerManager(
	client DockerInterface,
	recorder record.EventRecorder,
	prober prober.Prober,
	containerRefManager *kubecontainer.RefManager,
	machineInfo *cadvisorApi.MachineInfo,
	podInfraContainerImage string,
	qps float32,
	burst int,
	containerLogsDir string,
	osInterface kubecontainer.OSInterface,
	networkPlugin network.NetworkPlugin,
	generator kubecontainer.RunContainerOptionsGenerator,
	httpClient kubetypes.HttpGetter,
	execHandler ExecHandler,
	oomAdjuster *oom.OOMAdjuster,
	procFs procfs.ProcFsInterface,
	cpuCFSQuota bool,
	imageBackOff *util.Backoff) *DockerManager {

	// Work out the location of the Docker runtime, defaulting to /var/lib/docker
	// if there are any problems.
	dockerRoot := "/var/lib/docker"
	dockerInfo, err := client.Info()
	if err != nil {
		glog.Errorf("Failed to execute Info() call to the Docker client: %v", err)
		glog.Warningf("Using fallback default of /var/lib/docker for location of Docker runtime")
	} else {
		driverStatus := dockerInfo.Get("DriverStatus")
		// The DriverStatus is a*string* which represents a list of list of strings (pairs) e.g.
		// DriverStatus=[["Root Dir","/var/lib/docker/aufs"],["Backing Filesystem","extfs"],["Dirs","279"]]
		// Strip out the square brakcets and quotes.
		s := strings.Replace(driverStatus, "[", "", -1)
		s = strings.Replace(s, "]", "", -1)
		s = strings.Replace(s, `"`, "", -1)
		// Separate by commas.
		ss := strings.Split(s, ",")
		// Search for the Root Dir string
		for i, k := range ss {
			if k == "Root Dir" && i+1 < len(ss) {
				// Discard the /aufs suffix.
				dockerRoot, _ = path.Split(ss[i+1])
				// Trim the last slash.
				dockerRoot = strings.TrimSuffix(dockerRoot, "/")
				glog.Infof("Setting dockerRoot to %s", dockerRoot)
			}

		}
	}

	reasonCache := reasonInfoCache{cache: lru.New(maxReasonCacheEntries)}

	dm := &DockerManager{
		client:                 client,
		recorder:               recorder,
		containerRefManager:    containerRefManager,
		os:                     osInterface,
		machineInfo:            machineInfo,
		podInfraContainerImage: podInfraContainerImage,
		reasonCache:            reasonCache,
		dockerPuller:           newDockerPuller(client, qps, burst),
		dockerRoot:             dockerRoot,
		containerLogsDir:       containerLogsDir,
		networkPlugin:          networkPlugin,
		prober:                 prober,
		generator:              generator,
		execHandler:            execHandler,
		oomAdjuster:            oomAdjuster,
		procFs:                 procFs,
		cpuCFSQuota:            cpuCFSQuota,
	}
	dm.runner = lifecycle.NewHandlerRunner(httpClient, dm, dm)
	dm.imagePuller = kubecontainer.NewImagePuller(recorder, dm, imageBackOff)
	dm.containerGC = NewContainerGC(client, containerLogsDir)

	return dm
}

// A cache which stores strings keyed by <pod_UID>_<container_name>.
type reasonInfoCache struct {
	lock  sync.RWMutex
	cache *lru.Cache
}
type reasonInfo struct {
	reason  string
	message string
}

func (sc *reasonInfoCache) composeKey(uid types.UID, name string) string {
	return fmt.Sprintf("%s_%s", uid, name)
}

func (sc *reasonInfoCache) Add(uid types.UID, name string, reason, message string) {
	sc.lock.Lock()
	defer sc.lock.Unlock()
	sc.cache.Add(sc.composeKey(uid, name), reasonInfo{reason, message})
}

func (sc *reasonInfoCache) Remove(uid types.UID, name string) {
	sc.lock.Lock()
	defer sc.lock.Unlock()
	sc.cache.Remove(sc.composeKey(uid, name))
}

func (sc *reasonInfoCache) Get(uid types.UID, name string) (reasonInfo, bool) {
	sc.lock.RLock()
	defer sc.lock.RUnlock()
	value, ok := sc.cache.Get(sc.composeKey(uid, name))
	if ok {
		return value.(reasonInfo), ok
	} else {
		return reasonInfo{"", ""}, ok
	}
}

// GetContainerLogs returns logs of a specific container. By
// default, it returns a snapshot of the container log. Set 'follow' to true to
// stream the log. Set 'follow' to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
// TODO: Make 'RawTerminal' option  flagable.
func (dm *DockerManager) GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error) {
	var since int64
	if logOptions.SinceSeconds != nil {
		t := unversioned.Now().Add(-time.Duration(*logOptions.SinceSeconds) * time.Second)
		since = t.Unix()
	}
	if logOptions.SinceTime != nil {
		since = logOptions.SinceTime.Unix()
	}
	opts := docker.LogsOptions{
		Container:    containerID.ID,
		Stdout:       true,
		Stderr:       true,
		OutputStream: stdout,
		ErrorStream:  stderr,
		Timestamps:   logOptions.Timestamps,
		Since:        since,
		Follow:       logOptions.Follow,
		RawTerminal:  false,
	}

	if !logOptions.Follow && logOptions.TailLines != nil {
		opts.Tail = strconv.FormatInt(*logOptions.TailLines, 10)
	}

	err = dm.client.Logs(opts)
	return
}

var (
	// ErrNoContainersInPod is returned when there are no containers for a given pod
	ErrNoContainersInPod = errors.New("NoContainersInPod")

	// ErrNoPodInfraContainerInPod is returned when there is no pod infra container for a given pod
	ErrNoPodInfraContainerInPod = errors.New("NoPodInfraContainerInPod")

	// ErrContainerCannotRun is returned when a container is created, but cannot run properly
	ErrContainerCannotRun = errors.New("ContainerCannotRun")
)

// Internal information kept for containers from inspection
type containerStatusResult struct {
	status api.ContainerStatus
	ip     string
	err    error
}

func (dm *DockerManager) inspectContainer(dockerID, containerName, tPath string, pod *api.Pod) *containerStatusResult {
	result := containerStatusResult{api.ContainerStatus{}, "", nil}

	inspectResult, err := dm.client.InspectContainer(dockerID)

	if err != nil {
		result.err = err
		return &result
	}
	if inspectResult == nil {
		// Why did we not get an error?
		return &result
	}

	glog.V(4).Infof("Container inspect result: %+v", *inspectResult)
	result.status = api.ContainerStatus{
		Name:        containerName,
		Image:       inspectResult.Config.Image,
		ImageID:     DockerPrefix + inspectResult.Image,
		ContainerID: DockerPrefix + dockerID,
	}

	if inspectResult.State.Running {
		result.status.State.Running = &api.ContainerStateRunning{
			StartedAt: unversioned.NewTime(inspectResult.State.StartedAt),
		}
		if containerName == PodInfraContainerName {
			if inspectResult.NetworkSettings != nil {
				result.ip = inspectResult.NetworkSettings.IPAddress
			}
			// override the above if a network plugin exists
			if dm.networkPlugin.Name() != network.DefaultPluginName {
				netStatus, err := dm.networkPlugin.Status(pod.Namespace, pod.Name, kubetypes.DockerID(dockerID))
				if err != nil {
					glog.Errorf("NetworkPlugin %s failed on the status hook for pod '%s' - %v", dm.networkPlugin.Name(), pod.Name, err)
				} else if netStatus != nil {
					result.ip = netStatus.IP.String()
				}
			}
		}
	} else if !inspectResult.State.FinishedAt.IsZero() {
		reason := ""
		message := ""
		// Note: An application might handle OOMKilled gracefully.
		// In that case, the container is oom killed, but the exit
		// code could be 0.
		if inspectResult.State.OOMKilled {
			reason = "OOMKilled"
		} else {
			reason = "Error"
			message = inspectResult.State.Error
		}
		result.status.State.Terminated = &api.ContainerStateTerminated{
			ExitCode: inspectResult.State.ExitCode,
			Message:  message,
			Reason:   reason,

			StartedAt:   unversioned.NewTime(inspectResult.State.StartedAt),
			FinishedAt:  unversioned.NewTime(inspectResult.State.FinishedAt),
			ContainerID: DockerPrefix + dockerID,
		}
		if tPath != "" {
			path, found := inspectResult.Volumes[tPath]
			if found {
				data, err := ioutil.ReadFile(path)
				if err != nil {
					result.status.State.Terminated.Message = fmt.Sprintf("Error on reading termination-log %s: %v", path, err)
				} else {
					result.status.State.Terminated.Message = string(data)
				}
			}
		}
	} else {
		// TODO(dchen1107): Separate issue docker/docker#8294 was filed
		result.status.State.Waiting = &api.ContainerStateWaiting{
			Reason: ErrContainerCannotRun.Error(),
		}
	}

	return &result
}

// GetPodStatus returns docker related status for all containers in the pod as
// well as the infrastructure container.
func (dm *DockerManager) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID
	manifest := pod.Spec

	oldStatuses := make(map[string]api.ContainerStatus, len(pod.Spec.Containers))
	lastObservedTime := make(map[string]unversioned.Time, len(pod.Spec.Containers))
	// Record the last time we observed a container termination.
	for _, status := range pod.Status.ContainerStatuses {
		oldStatuses[status.Name] = status
		if status.LastTerminationState.Terminated != nil {
			timestamp, ok := lastObservedTime[status.Name]
			if !ok || timestamp.Before(status.LastTerminationState.Terminated.FinishedAt) {
				lastObservedTime[status.Name] = status.LastTerminationState.Terminated.FinishedAt
			}
		}
	}

	var podStatus api.PodStatus
	statuses := make(map[string]*api.ContainerStatus, len(pod.Spec.Containers))

	expectedContainers := make(map[string]api.Container)
	for _, container := range manifest.Containers {
		expectedContainers[container.Name] = container
	}
	expectedContainers[PodInfraContainerName] = api.Container{}

	containers, err := dm.client.ListContainers(docker.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}

	containerDone := sets.NewString()
	// Loop through list of running and exited docker containers to construct
	// the statuses. We assume docker returns a list of containers sorted in
	// reverse by time.
	for _, value := range containers {
		if len(value.Names) == 0 {
			continue
		}
		dockerName, _, err := ParseDockerName(value.Names[0])
		if err != nil {
			continue
		}
		if dockerName.PodFullName != podFullName {
			continue
		}
		if uid != "" && dockerName.PodUID != uid {
			continue
		}
		dockerContainerName := dockerName.ContainerName
		c, found := expectedContainers[dockerContainerName]
		if !found {
			continue
		}
		terminationMessagePath := c.TerminationMessagePath
		if containerDone.Has(dockerContainerName) {
			continue
		}

		var terminationState *api.ContainerState = nil
		// Inspect the container.
		result := dm.inspectContainer(value.ID, dockerContainerName, terminationMessagePath, pod)
		if result.err != nil {
			return nil, result.err
		} else if result.status.State.Terminated != nil {
			terminationState = &result.status.State
		}

		if containerStatus, found := statuses[dockerContainerName]; found {
			if containerStatus.LastTerminationState.Terminated == nil && terminationState != nil {
				// Populate the last termination state.
				containerStatus.LastTerminationState = *terminationState
			}
			if terminationState == nil {
				// Not a dead container.
				continue
			}
			// Only count dead containers terminated after last time we observed,
			lastObservedTime, ok := lastObservedTime[dockerContainerName]
			if !ok || terminationState.Terminated.FinishedAt.After(lastObservedTime.Time) {
				containerStatus.RestartCount += 1
			} else {
				// The container finished before the last observation. No
				// need to examine/count the older containers. Mark the
				// container name as done.
				containerDone.Insert(dockerContainerName)
			}
			continue
		}

		if dockerContainerName == PodInfraContainerName {
			// Found network container
			if result.status.State.Running != nil {
				podStatus.PodIP = result.ip
			}
		} else {
			// Add user container information.
			if oldStatus, found := oldStatuses[dockerContainerName]; found {
				// Use the last observed restart count if it's available.
				result.status.RestartCount = oldStatus.RestartCount
			}
			statuses[dockerContainerName] = &result.status
		}
	}

	// Handle the containers for which we cannot find any associated active or dead docker containers or are in restart backoff
	for _, container := range manifest.Containers {
		if containerStatus, found := statuses[container.Name]; found {
			reasonInfo, ok := dm.reasonCache.Get(uid, container.Name)
			if ok && reasonInfo.reason == kubecontainer.ErrCrashLoopBackOff.Error() {
				// We need to increment the restart count if we are going to
				// move the current state to last terminated state.
				if containerStatus.State.Terminated != nil {
					lastObservedTime, ok := lastObservedTime[container.Name]
					if !ok || containerStatus.State.Terminated.FinishedAt.After(lastObservedTime.Time) {
						containerStatus.RestartCount += 1
					}
				}
				containerStatus.LastTerminationState = containerStatus.State
				containerStatus.State = api.ContainerState{
					Waiting: &api.ContainerStateWaiting{
						Reason:  reasonInfo.reason,
						Message: reasonInfo.message,
					},
				}
			}
			continue
		}
		var containerStatus api.ContainerStatus
		containerStatus.Name = container.Name
		containerStatus.Image = container.Image
		if oldStatus, found := oldStatuses[container.Name]; found {
			// Some states may be lost due to GC; apply the last observed
			// values if possible.
			containerStatus.RestartCount = oldStatus.RestartCount
			containerStatus.LastTerminationState = oldStatus.LastTerminationState
		}
		// TODO(dchen1107): docker/docker/issues/8365 to figure out if the image exists
		reasonInfo, ok := dm.reasonCache.Get(uid, container.Name)
		if !ok {
			// default position for a container
			// At this point there are no active or dead containers, the reasonCache is empty (no entry or the entry has expired)
			// its reasonable to say the container is being created till a more accurate reason is logged
			containerStatus.State = api.ContainerState{
				Waiting: &api.ContainerStateWaiting{
					Reason:  fmt.Sprintf("ContainerCreating"),
					Message: fmt.Sprintf("Image: %s is ready, container is creating", container.Image),
				},
			}
		} else if reasonInfo.reason == kubecontainer.ErrImagePullBackOff.Error() ||
			reasonInfo.reason == kubecontainer.ErrImageInspect.Error() ||
			reasonInfo.reason == kubecontainer.ErrImagePull.Error() ||
			reasonInfo.reason == kubecontainer.ErrImageNeverPull.Error() {
			// mark it as waiting, reason will be filled bellow
			containerStatus.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
		} else if reasonInfo.reason == kubecontainer.ErrRunContainer.Error() {
			// mark it as waiting, reason will be filled bellow
			containerStatus.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
		}
		statuses[container.Name] = &containerStatus
	}

	podStatus.ContainerStatuses = make([]api.ContainerStatus, 0)
	for containerName, status := range statuses {
		if status.State.Waiting != nil {
			status.State.Running = nil
			// For containers in the waiting state, fill in a specific reason if it is recorded.
			if reasonInfo, ok := dm.reasonCache.Get(uid, containerName); ok {
				status.State.Waiting.Reason = reasonInfo.reason
				status.State.Waiting.Message = reasonInfo.message
			}
		}
		podStatus.ContainerStatuses = append(podStatus.ContainerStatuses, *status)
	}
	// Sort the container statuses since clients of this interface expect the list
	// of containers in a pod to behave like the output of `docker list`, which has a
	// deterministic order.
	sort.Sort(kubetypes.SortedContainerStatuses(podStatus.ContainerStatuses))
	return &podStatus, nil
}

// makeEnvList converts EnvVar list to a list of strings, in the form of
// '<key>=<value>', which can be understood by docker.
func makeEnvList(envs []kubecontainer.EnvVar) (result []string) {
	for _, env := range envs {
		result = append(result, fmt.Sprintf("%s=%s", env.Name, env.Value))
	}
	return
}

// makeMountBindings converts the mount list to a list of strings that
// can be understood by docker.
// Each element in the string is in the form of:
// '<HostPath>:<ContainerPath>', or
// '<HostPath>:<ContainerPath>:ro', if the path is read only.
func makeMountBindings(mounts []kubecontainer.Mount) (result []string) {
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.HostPath, m.ContainerPath)
		if m.ReadOnly {
			bind += ":ro"
		}
		result = append(result, bind)
	}
	return
}

func makePortsAndBindings(portMappings []kubecontainer.PortMapping) (map[docker.Port]struct{}, map[docker.Port][]docker.PortBinding) {
	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range portMappings {
		exteriorPort := port.HostPort
		if exteriorPort == 0 {
			// No need to do port binding when HostPort is not specified
			continue
		}
		interiorPort := port.ContainerPort
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		var protocol string
		switch strings.ToUpper(string(port.Protocol)) {
		case "UDP":
			protocol = "/udp"
		case "TCP":
			protocol = "/tcp"
		default:
			glog.Warningf("Unknown protocol %q: defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}

		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}

		hostBinding := docker.PortBinding{
			HostPort: strconv.Itoa(exteriorPort),
			HostIP:   port.HostIP,
		}

		// Allow multiple host ports bind to same docker port
		if existedBindings, ok := portBindings[dockerPort]; ok {
			// If a docker port already map to a host port, just append the host ports
			portBindings[dockerPort] = append(existedBindings, hostBinding)
		} else {
			// Otherwise, it's fresh new port binding
			portBindings[dockerPort] = []docker.PortBinding{
				hostBinding,
			}
		}
	}
	return exposedPorts, portBindings
}

func (dm *DockerManager) runContainer(
	pod *api.Pod,
	container *api.Container,
	opts *kubecontainer.RunContainerOptions,
	ref *api.ObjectReference,
	netMode string,
	ipcMode string,
	utsMode string,
	pidMode string) (kubecontainer.ContainerID, error) {

	dockerName := KubeletContainerName{
		PodFullName:   kubecontainer.GetPodFullName(pod),
		PodUID:        pod.UID,
		ContainerName: container.Name,
	}
	exposedPorts, portBindings := makePortsAndBindings(opts.PortMappings)

	// TODO(vmarmol): Handle better.
	// Cap hostname at 63 chars (specification is 64bytes which is 63 chars and the null terminating char).
	const hostnameMaxLen = 63
	containerHostname := pod.Name
	if len(containerHostname) > hostnameMaxLen {
		containerHostname = containerHostname[:hostnameMaxLen]
	}

	// Pod information is recorded on the container as labels to preserve it in the event the pod is deleted
	// while the Kubelet is down and there is no information available to recover the pod. This includes
	// termination information like the termination grace period and the pre stop hooks.
	// TODO: keep these labels up to date if the pod changes
	namespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	labels := map[string]string{
		kubernetesNameLabel: namespacedName.String(),
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		labels[kubernetesTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}
	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// TODO: This is kind of hacky, we should really just encode the bits we need.
		data, err := latest.GroupOrDie("").Codec.Encode(pod)
		if err != nil {
			glog.Errorf("Failed to encode pod: %s for prestop hook", pod.Name)
		} else {
			labels[kubernetesPodLabel] = string(data)
			labels[kubernetesContainerLabel] = container.Name
		}
	}
	memoryLimit := container.Resources.Limits.Memory().Value()
	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	var cpuShares int64
	// If request is not specified, but limit is, we want request to default to limit.
	// API server does this for new containers, but we repeat this logic in Kubelet
	// for containers running on existing Kubernetes clusters.
	if cpuRequest.Amount == nil && cpuLimit.Amount != nil {
		cpuShares = milliCPUToShares(cpuLimit.MilliValue())
	} else {
		// if cpuRequest.Amount is nil, then milliCPUToShares will return the minimal number
		// of CPU shares.
		cpuShares = milliCPUToShares(cpuRequest.MilliValue())
	}

	_, containerName := BuildDockerName(dockerName, container)
	dockerOpts := docker.CreateContainerOptions{
		Name: containerName,
		Config: &docker.Config{
			Env:          makeEnvList(opts.Envs),
			ExposedPorts: exposedPorts,
			Hostname:     containerHostname,
			Image:        container.Image,
			// Memory and CPU are set here for older versions of Docker (pre-1.6).
			Memory:     memoryLimit,
			MemorySwap: -1,
			CPUShares:  cpuShares,
			WorkingDir: container.WorkingDir,
			Labels:     labels,
			// Interactive containers:
			OpenStdin: container.Stdin,
			Tty:       container.TTY,
		},
	}

	setEntrypointAndCommand(container, opts, &dockerOpts)

	glog.V(3).Infof("Container %v/%v/%v: setting entrypoint \"%v\" and command \"%v\"", pod.Namespace, pod.Name, container.Name, dockerOpts.Config.Entrypoint, dockerOpts.Config.Cmd)

	securityContextProvider := securitycontext.NewSimpleSecurityContextProvider()
	securityContextProvider.ModifyContainerConfig(pod, container, dockerOpts.Config)
	dockerContainer, err := dm.client.CreateContainer(dockerOpts)
	if err != nil {
		if ref != nil {
			dm.recorder.Eventf(ref, "Failed", "Failed to create docker container with error: %v", err)
		}
		return kubecontainer.ContainerID{}, err
	}

	if ref != nil {
		dm.recorder.Eventf(ref, "Created", "Created with docker id %v", util.ShortenString(dockerContainer.ID, 12))
	}

	binds := makeMountBindings(opts.Mounts)

	// The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
	// TODO(yifan): Consider to pull this logic out since we might need to reuse it in
	// other container runtime.
	if opts.PodContainerDir != "" && len(container.TerminationMessagePath) != 0 {
		containerLogPath := path.Join(opts.PodContainerDir, dockerContainer.ID)
		fs, err := os.Create(containerLogPath)
		if err != nil {
			// TODO: Clean up the previouly created dir? return the error?
			glog.Errorf("Error on creating termination-log file %q: %v", containerLogPath, err)
		} else {
			fs.Close() // Close immediately; we're just doing a `touch` here
			b := fmt.Sprintf("%s:%s", containerLogPath, container.TerminationMessagePath)
			binds = append(binds, b)
		}
	}

	hc := &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
		IpcMode:      ipcMode,
		UTSMode:      utsMode,
		PidMode:      pidMode,
		// Memory and CPU are set here for newer versions of Docker (1.6+).
		Memory:     memoryLimit,
		MemorySwap: -1,
		CPUShares:  cpuShares,
	}

	if dm.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned to allow full usage of cpu resource.
		cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())

		hc.CPUQuota = cpuQuota
		hc.CPUPeriod = cpuPeriod
	}

	if len(opts.DNS) > 0 {
		hc.DNS = opts.DNS
	}
	if len(opts.DNSSearch) > 0 {
		hc.DNSSearch = opts.DNSSearch
	}
	if len(opts.CgroupParent) > 0 {
		hc.CgroupParent = opts.CgroupParent
	}
	securityContextProvider.ModifyHostConfig(pod, container, hc)

	if err = dm.client.StartContainer(dockerContainer.ID, hc); err != nil {
		if ref != nil {
			dm.recorder.Eventf(ref, "Failed",
				"Failed to start with docker id %v with error: %v", util.ShortenString(dockerContainer.ID, 12), err)
		}
		return kubecontainer.ContainerID{}, err
	}
	if ref != nil {
		dm.recorder.Eventf(ref, "Started", "Started with docker id %v", util.ShortenString(dockerContainer.ID, 12))
	}
	return kubetypes.DockerID(dockerContainer.ID).ContainerID(), nil
}

func setEntrypointAndCommand(container *api.Container, opts *kubecontainer.RunContainerOptions, dockerOpts *docker.CreateContainerOptions) {
	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)

	dockerOpts.Config.Entrypoint = command
	dockerOpts.Config.Cmd = args
}

// A helper function to get the KubeletContainerName and hash from a docker
// container.
func getDockerContainerNameInfo(c *docker.APIContainers) (*KubeletContainerName, uint64, error) {
	if len(c.Names) == 0 {
		return nil, 0, fmt.Errorf("cannot parse empty docker container name: %#v", c.Names)
	}
	dockerName, hash, err := ParseDockerName(c.Names[0])
	if err != nil {
		return nil, 0, fmt.Errorf("parse docker container name %q error: %v", c.Names[0], err)
	}
	return dockerName, hash, nil
}

// Get pod UID, name, and namespace by examining the container names.
func getPodInfoFromContainer(c *docker.APIContainers) (types.UID, string, string, error) {
	dockerName, _, err := getDockerContainerNameInfo(c)
	if err != nil {
		return types.UID(""), "", "", err
	}
	name, namespace, err := kubecontainer.ParsePodFullName(dockerName.PodFullName)
	if err != nil {
		return types.UID(""), "", "", fmt.Errorf("parse pod full name %q error: %v", dockerName.PodFullName, err)
	}
	return dockerName.PodUID, name, namespace, nil
}

// GetContainers returns a list of running containers if |all| is false;
// otherwise, it returns all containers.
func (dm *DockerManager) GetContainers(all bool) ([]*kubecontainer.Container, error) {
	containers, err := GetKubeletDockerContainers(dm.client, all)
	if err != nil {
		return nil, err
	}
	// Convert DockerContainers to []*kubecontainer.Container
	result := make([]*kubecontainer.Container, 0, len(containers))
	for _, c := range containers {
		converted, err := toRuntimeContainer(c)
		if err != nil {
			glog.Errorf("Error examining the container: %v", err)
			continue
		}
		result = append(result, converted)
	}
	return result, nil
}

func (dm *DockerManager) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("GetPods").Observe(metrics.SinceInMicroseconds(start))
	}()
	pods := make(map[types.UID]*kubecontainer.Pod)
	var result []*kubecontainer.Pod

	containers, err := GetKubeletDockerContainers(dm.client, all)
	if err != nil {
		return nil, err
	}

	// Group containers by pod.
	for _, c := range containers {
		converted, err := toRuntimeContainer(c)
		if err != nil {
			glog.Errorf("Error examining the container: %v", err)
			continue
		}

		podUID, podName, podNamespace, err := getPodInfoFromContainer(c)
		if err != nil {
			glog.Errorf("Error examining the container: %v", err)
			continue
		}

		pod, found := pods[podUID]
		if !found {
			pod = &kubecontainer.Pod{
				ID:        podUID,
				Name:      podName,
				Namespace: podNamespace,
			}
			pods[podUID] = pod
		}
		pod.Containers = append(pod.Containers, converted)
	}

	// Convert map to list.
	for _, c := range pods {
		result = append(result, c)
	}
	return result, nil
}

// List all images in the local storage.
func (dm *DockerManager) ListImages() ([]kubecontainer.Image, error) {
	var images []kubecontainer.Image

	dockerImages, err := dm.client.ListImages(docker.ListImagesOptions{})
	if err != nil {
		return images, err
	}

	for _, di := range dockerImages {
		image, err := toRuntimeImage(&di)
		if err != nil {
			continue
		}
		images = append(images, *image)
	}
	return images, nil
}

// TODO(vmarmol): Consider unexporting.
// PullImage pulls an image from network to local storage.
func (dm *DockerManager) PullImage(image kubecontainer.ImageSpec, secrets []api.Secret) error {
	return dm.dockerPuller.Pull(image.Image, secrets)
}

// IsImagePresent checks whether the container image is already in the local storage.
func (dm *DockerManager) IsImagePresent(image kubecontainer.ImageSpec) (bool, error) {
	return dm.dockerPuller.IsImagePresent(image.Image)
}

// Removes the specified image.
func (dm *DockerManager) RemoveImage(image kubecontainer.ImageSpec) error {
	return dm.client.RemoveImage(image.Image)
}

// podInfraContainerChanged returns true if the pod infra container has changed.
func (dm *DockerManager) podInfraContainerChanged(pod *api.Pod, podInfraContainer *kubecontainer.Container) (bool, error) {
	networkMode := ""
	var ports []api.ContainerPort

	dockerPodInfraContainer, err := dm.client.InspectContainer(podInfraContainer.ID.ID)
	if err != nil {
		return false, err
	}

	// Check network mode.
	if dockerPodInfraContainer.HostConfig != nil {
		networkMode = dockerPodInfraContainer.HostConfig.NetworkMode
	}
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork {
		if networkMode != "host" {
			glog.V(4).Infof("host: %v, %v", pod.Spec.SecurityContext.HostNetwork, networkMode)
			return true, nil
		}
	} else {
		// Docker only exports ports from the pod infra container. Let's
		// collect all of the relevant ports and export them.
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}
	expectedPodInfraContainer := &api.Container{
		Name:            PodInfraContainerName,
		Image:           dm.podInfraContainerImage,
		Ports:           ports,
		ImagePullPolicy: podInfraContainerImagePullPolicy,
	}
	return podInfraContainer.Hash != kubecontainer.HashContainer(expectedPodInfraContainer), nil
}

type dockerVersion docker.APIVersion

func NewVersion(input string) (dockerVersion, error) {
	version, err := docker.NewAPIVersion(input)
	return dockerVersion(version), err
}

func (dv dockerVersion) String() string {
	return docker.APIVersion(dv).String()
}

func (dv dockerVersion) Compare(other string) (int, error) {
	a := docker.APIVersion(dv)
	b, err := docker.NewAPIVersion(other)
	if err != nil {
		return 0, err
	}
	if a.LessThan(b) {
		return -1, nil
	}
	if a.GreaterThan(b) {
		return 1, nil
	}
	return 0, nil
}

func (dm *DockerManager) Version() (kubecontainer.Version, error) {
	env, err := dm.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}

	apiVersion := env.Get("ApiVersion")
	version, err := docker.NewAPIVersion(apiVersion)
	if err != nil {
		glog.Errorf("docker: failed to parse docker server version %q: %v", apiVersion, err)
		return nil, fmt.Errorf("docker: failed to parse docker server version %q: %v", apiVersion, err)
	}
	return dockerVersion(version), nil
}

// The first version of docker that supports exec natively is 1.3.0 == API 1.15
var dockerAPIVersionWithExec = "1.15"

func (dm *DockerManager) nativeExecSupportExists() (bool, error) {
	version, err := dm.Version()
	if err != nil {
		return false, err
	}
	result, err := version.Compare(dockerAPIVersionWithExec)
	if result >= 0 {
		return true, err
	}
	return false, err
}

func (dm *DockerManager) getRunInContainerCommand(containerID kubecontainer.ContainerID, cmd []string) (*exec.Cmd, error) {
	args := append([]string{"exec"}, cmd...)
	command := exec.Command("/usr/sbin/nsinit", args...)
	command.Dir = fmt.Sprintf("/var/lib/docker/execdriver/native/%s", containerID.ID)
	return command, nil
}

func (dm *DockerManager) runInContainerUsingNsinit(containerID kubecontainer.ContainerID, cmd []string) ([]byte, error) {
	c, err := dm.getRunInContainerCommand(containerID, cmd)
	if err != nil {
		return nil, err
	}
	return c.CombinedOutput()
}

// RunInContainer uses nsinit to run the command inside the container identified by containerID
func (dm *DockerManager) RunInContainer(containerID kubecontainer.ContainerID, cmd []string) ([]byte, error) {
	// If native exec support does not exist in the local docker daemon use nsinit.
	useNativeExec, err := dm.nativeExecSupportExists()
	if err != nil {
		return nil, err
	}
	if !useNativeExec {
		glog.V(2).Infof("Using nsinit to run the command %+v inside container %s", cmd, containerID)
		return dm.runInContainerUsingNsinit(containerID, cmd)
	}
	glog.V(2).Infof("Using docker native exec to run cmd %+v inside container %s", cmd, containerID)
	createOpts := docker.CreateExecOptions{
		Container:    containerID.ID,
		Cmd:          cmd,
		AttachStdin:  false,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
	}
	execObj, err := dm.client.CreateExec(createOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to run in container - Exec setup failed - %v", err)
	}
	var buf bytes.Buffer
	startOpts := docker.StartExecOptions{
		Detach:       false,
		Tty:          false,
		OutputStream: &buf,
		ErrorStream:  &buf,
		RawTerminal:  false,
	}
	err = dm.client.StartExec(execObj.ID, startOpts)
	if err != nil {
		glog.V(2).Infof("StartExec With error: %v", err)
		return nil, err
	}
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		inspect, err2 := dm.client.InspectExec(execObj.ID)
		if err2 != nil {
			glog.V(2).Infof("InspectExec %s failed with error: %+v", execObj.ID, err2)
			return buf.Bytes(), err2
		}
		if !inspect.Running {
			if inspect.ExitCode != 0 {
				glog.V(2).Infof("InspectExec %s exit with result %+v", execObj.ID, inspect)
				err = &dockerExitError{inspect}
			}
			break
		}
		<-ticker.C
	}

	return buf.Bytes(), err
}

type dockerExitError struct {
	Inspect *docker.ExecInspect
}

func (d *dockerExitError) String() string {
	return d.Error()
}

func (d *dockerExitError) Error() string {
	return fmt.Sprintf("Error executing in Docker Container: %d", d.Inspect.ExitCode)
}

func (d *dockerExitError) Exited() bool {
	return !d.Inspect.Running
}

func (d *dockerExitError) ExitStatus() int {
	return d.Inspect.ExitCode
}

// ExecInContainer runs the command inside the container identified by containerID.
func (dm *DockerManager) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	if dm.execHandler == nil {
		return errors.New("unable to exec without an exec handler")
	}

	container, err := dm.client.InspectContainer(containerID.ID)
	if err != nil {
		return err
	}
	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container.ID)
	}

	return dm.execHandler.ExecInContainer(dm.client, container, cmd, stdin, stdout, stderr, tty)
}

func (dm *DockerManager) AttachContainer(containerID kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	opts := docker.AttachToContainerOptions{
		Container:    containerID.ID,
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		Stream:       true,
		Logs:         true,
		Stdin:        stdin != nil,
		Stdout:       stdout != nil,
		Stderr:       stderr != nil,
		RawTerminal:  tty,
	}
	return dm.client.AttachToContainer(opts)
}

func noPodInfraContainerError(podName, podNamespace string) error {
	return fmt.Errorf("cannot find pod infra container in pod %q", kubecontainer.BuildPodFullName(podName, podNamespace))
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
func (dm *DockerManager) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	podInfraContainer := pod.FindContainerByName(PodInfraContainerName)
	if podInfraContainer == nil {
		return noPodInfraContainerError(pod.Name, pod.Namespace)
	}
	container, err := dm.client.InspectContainer(podInfraContainer.ID.ID)
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container.ID)
	}

	containerPid := container.State.Pid
	socatPath, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}

	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-n", socatPath, "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	nsenterPath, lookupErr := exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}

	command := exec.Command(nsenterPath, args...)
	command.Stdout = stream

	// If we use Stdin, command.Run() won't return until the goroutine that's copying
	// from stream finishes. Unfortunately, if you have a client like telnet connected
	// via port forwarding, as long as the user's telnet client is connected to the user's
	// local listener that port forwarding sets up, the telnet session never exits. This
	// means that even if socat has finished running, command.Run() won't ever return
	// (because the client still has the connection and stream open).
	//
	// The work around is to use StdinPipe(), as Wait() (called by Run()) closes the pipe
	// when the command (socat) exits.
	inPipe, err := command.StdinPipe()
	if err != nil {
		return fmt.Errorf("unable to do port forwarding: error creating stdin pipe: %v", err)
	}
	go func() {
		io.Copy(inPipe, stream)
		inPipe.Close()
	}()

	return command.Run()
}

// Get the IP address of a container's interface using nsenter
func (dm *DockerManager) GetContainerIP(containerID, interfaceName string) (string, error) {
	_, lookupErr := exec.LookPath("nsenter")
	if lookupErr != nil {
		return "", fmt.Errorf("Unable to obtain IP address of container: missing nsenter.")
	}
	container, err := dm.client.InspectContainer(containerID)
	if err != nil {
		return "", err
	}

	if !container.State.Running {
		return "", fmt.Errorf("container not running (%s)", container.ID)
	}

	containerPid := container.State.Pid
	extractIPCmd := fmt.Sprintf("ip -4 addr show %s | grep inet | awk -F\" \" '{print $2}'", interfaceName)
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-n", "--", "bash", "-c", extractIPCmd}
	command := exec.Command("nsenter", args...)
	out, err := command.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// Kills all containers in the specified pod
func (dm *DockerManager) KillPod(pod *api.Pod, runningPod kubecontainer.Pod) error {
	// Send the kills in parallel since they may take a long time. Len + 1 since there
	// can be Len errors + the networkPlugin teardown error.
	errs := make(chan error, len(runningPod.Containers)+1)
	wg := sync.WaitGroup{}
	var (
		networkContainer *kubecontainer.Container
		networkSpec      *api.Container
	)
	for _, container := range runningPod.Containers {
		wg.Add(1)
		go func(container *kubecontainer.Container) {
			defer util.HandleCrash()
			defer wg.Done()

			var containerSpec *api.Container
			if pod != nil {
				for i, c := range pod.Spec.Containers {
					if c.Name == container.Name {
						containerSpec = &pod.Spec.Containers[i]
						break
					}
				}
			}

			// TODO: Handle this without signaling the pod infra container to
			// adapt to the generic container runtime.
			if container.Name == PodInfraContainerName {
				// Store the container runtime for later deletion.
				// We do this so that PreStop handlers can run in the network namespace.
				networkContainer = container
				networkSpec = containerSpec
				return
			}

			err := dm.KillContainerInPod(container.ID, containerSpec, pod)
			if err != nil {
				glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, runningPod.ID)
				errs <- err
			}
		}(container)
	}
	wg.Wait()
	if networkContainer != nil {
		if err := dm.networkPlugin.TearDownPod(runningPod.Namespace, runningPod.Name, kubetypes.DockerID(networkContainer.ID.ID)); err != nil {
			glog.Errorf("Failed tearing down the infra container: %v", err)
			errs <- err
		}
		if err := dm.KillContainerInPod(networkContainer.ID, networkSpec, pod); err != nil {
			glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, runningPod.ID)
			errs <- err
		}
	}
	close(errs)
	if len(errs) > 0 {
		errList := []error{}
		for err := range errs {
			errList = append(errList, err)
		}
		return fmt.Errorf("failed to delete containers (%v)", errList)
	}
	return nil
}

// KillContainerInPod kills a container in the pod. It must be passed either a container ID or a container and pod,
// and will attempt to lookup the other information if missing.
func (dm *DockerManager) KillContainerInPod(containerID kubecontainer.ContainerID, container *api.Container, pod *api.Pod) error {
	switch {
	case containerID.IsEmpty():
		// Locate the container.
		pods, err := dm.GetPods(false)
		if err != nil {
			return err
		}
		targetPod := kubecontainer.Pods(pods).FindPod(kubecontainer.GetPodFullName(pod), pod.UID)
		targetContainer := targetPod.FindContainerByName(container.Name)
		if targetContainer == nil {
			return fmt.Errorf("unable to find container %q in pod %q", container.Name, targetPod.Name)
		}
		containerID = targetContainer.ID

	case container == nil || pod == nil:
		// Read information about the container from labels
		inspect, err := dm.client.InspectContainer(containerID.ID)
		if err != nil {
			return err
		}
		storedPod, storedContainer, cerr := containerAndPodFromLabels(inspect)
		if cerr != nil {
			glog.Errorf("unable to access pod data from container: %v", err)
		}
		if container == nil {
			container = storedContainer
		}
		if pod == nil {
			pod = storedPod
		}
	}
	return dm.killContainer(containerID, container, pod)
}

// killContainer accepts a containerID and an optional container or pod containing shutdown policies. Invoke
// KillContainerInPod if information must be retrieved first.
func (dm *DockerManager) killContainer(containerID kubecontainer.ContainerID, container *api.Container, pod *api.Pod) error {
	ID := containerID.ID
	name := ID
	if container != nil {
		name = fmt.Sprintf("%s %s", name, container.Name)
	}
	if pod != nil {
		name = fmt.Sprintf("%s %s/%s", name, pod.Namespace, pod.Name)
	}

	gracePeriod := int64(minimumGracePeriodInSeconds)
	if pod != nil {
		switch {
		case pod.DeletionGracePeriodSeconds != nil:
			gracePeriod = *pod.DeletionGracePeriodSeconds
		case pod.Spec.TerminationGracePeriodSeconds != nil:
			gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
		}
	}
	glog.V(2).Infof("Killing container %q with %d second grace period", name, gracePeriod)
	start := unversioned.Now()

	if pod != nil && container != nil && container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		glog.V(4).Infof("Running preStop hook for container %q", name)
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer util.HandleCrash()
			if err := dm.runner.Run(containerID, pod, container, container.Lifecycle.PreStop); err != nil {
				glog.Errorf("preStop hook for container %q failed: %v", name, err)
			}
		}()
		select {
		case <-time.After(time.Duration(gracePeriod) * time.Second):
			glog.V(2).Infof("preStop hook for container %q did not complete in %d seconds", name, gracePeriod)
		case <-done:
			glog.V(4).Infof("preStop hook for container %q completed", name)
		}
		gracePeriod -= int64(unversioned.Now().Sub(start.Time).Seconds())
	}

	// always give containers a minimal shutdown window to avoid unnecessary SIGKILLs
	if gracePeriod < minimumGracePeriodInSeconds {
		gracePeriod = minimumGracePeriodInSeconds
	}
	err := dm.client.StopContainer(ID, uint(gracePeriod))
	if _, ok := err.(*docker.ContainerNotRunning); ok && err != nil {
		glog.V(4).Infof("Container %q has already exited", name)
		return nil
	}
	if err == nil {
		glog.V(2).Infof("Container %q exited after %s", name, unversioned.Now().Sub(start.Time))
	} else {
		glog.V(2).Infof("Container %q termination failed after %s: %v", name, unversioned.Now().Sub(start.Time), err)
	}
	ref, ok := dm.containerRefManager.GetRef(containerID)
	if !ok {
		glog.Warningf("No ref for pod '%q'", name)
	} else {
		// TODO: pass reason down here, and state, or move this call up the stack.
		dm.recorder.Eventf(ref, "Killing", "Killing with docker id %v", util.ShortenString(ID, 12))
		dm.containerRefManager.ClearRef(containerID)
	}
	return err
}

var errNoPodOnContainer = fmt.Errorf("no pod information labels on Docker container")

// containerAndPodFromLabels tries to load the appropriate container info off of a Docker container's labels
func containerAndPodFromLabels(inspect *docker.Container) (pod *api.Pod, container *api.Container, err error) {
	if inspect == nil && inspect.Config == nil && inspect.Config.Labels == nil {
		return nil, nil, errNoPodOnContainer
	}
	labels := inspect.Config.Labels

	// the pod data may not be set
	if body, found := labels[kubernetesPodLabel]; found {
		pod = &api.Pod{}
		if err = latest.GroupOrDie("").Codec.DecodeInto([]byte(body), pod); err == nil {
			name := labels[kubernetesContainerLabel]
			for ix := range pod.Spec.Containers {
				if pod.Spec.Containers[ix].Name == name {
					container = &pod.Spec.Containers[ix]
					break
				}
			}
			if container == nil {
				err = fmt.Errorf("unable to find container %s in pod %v", name, pod)
			}
		} else {
			pod = nil
		}
	}

	// attempt to find the default grace period if we didn't commit a pod, but set the generic metadata
	// field (the one used by kill)
	if pod == nil {
		if period, ok := labels[kubernetesTerminationGracePeriodLabel]; ok {
			if seconds, err := strconv.ParseInt(period, 10, 64); err == nil {
				pod = &api.Pod{}
				pod.DeletionGracePeriodSeconds = &seconds
			}
		}
	}

	return
}

// Run a single container from a pod. Returns the docker container ID
func (dm *DockerManager) runContainerInPod(pod *api.Pod, container *api.Container, netMode, ipcMode, pidMode string) (kubecontainer.ContainerID, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("runContainerInPod").Observe(metrics.SinceInMicroseconds(start))
	}()

	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	opts, err := dm.generator.GenerateRunContainerOptions(pod, container)
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}

	utsMode := ""
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork {
		utsMode = "host"
	}
	id, err := dm.runContainer(pod, container, opts, ref, netMode, ipcMode, utsMode, pidMode)
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}

	// Remember this reference so we can report events about this container
	if ref != nil {
		dm.containerRefManager.SetRef(id, ref)
	}

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := dm.runner.Run(id, pod, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			dm.KillContainerInPod(id, container, pod)
			return kubecontainer.ContainerID{}, fmt.Errorf("failed to call event handler: %v", handlerErr)
		}
	}

	// Create a symbolic link to the Docker container log file using a name which captures the
	// full pod name, the container name and the Docker container ID. Cluster level logging will
	// capture these symbolic filenames which can be used for search terms in Elasticsearch or for
	// labels for Cloud Logging.
	podFullName := kubecontainer.GetPodFullName(pod)
	containerLogFile := path.Join(dm.dockerRoot, "containers", id.ID, fmt.Sprintf("%s-json.log", id.ID))
	symlinkFile := LogSymlink(dm.containerLogsDir, podFullName, container.Name, id.ID)
	if err = dm.os.Symlink(containerLogFile, symlinkFile); err != nil {
		glog.Errorf("Failed to create symbolic link to the log file of pod %q container %q: %v", podFullName, container.Name, err)
	}

	// Container information is used in adjusting OOM scores and adding ndots.
	containerInfo, err := dm.client.InspectContainer(id.ID)
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}
	// Ensure the PID actually exists, else we'll move ourselves.
	if containerInfo.State.Pid == 0 {
		return kubecontainer.ContainerID{}, fmt.Errorf("failed to get init PID for Docker container %q", id)
	}

	// Set OOM score of the container based on the priority of the container.
	// Processes in lower-priority pods should be killed first if the system runs out of memory.
	// The main pod infrastructure container is considered high priority, since if it is killed the
	// whole pod will die.
	var oomScoreAdj int
	if container.Name == PodInfraContainerName {
		oomScoreAdj = qos.PodInfraOOMAdj
	} else {
		oomScoreAdj = qos.GetContainerOOMScoreAdjust(container, dm.machineInfo.MemoryCapacity)
	}
	cgroupName, err := dm.procFs.GetFullContainerName(containerInfo.State.Pid)
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}
	if err = dm.oomAdjuster.ApplyOOMScoreAdjContainer(cgroupName, oomScoreAdj, 5); err != nil {
		return kubecontainer.ContainerID{}, err
	}

	// currently, Docker does not have a flag by which the ndots option can be passed.
	// (A separate issue has been filed with Docker to add a ndots flag)
	// The addNDotsOption call appends the ndots option to the resolv.conf file generated by docker.
	// This resolv.conf file is shared by all containers of the same pod, and needs to be modified only once per pod.
	// we modify it when the pause container is created since it is the first container created in the pod since it holds
	// the networking namespace.
	if container.Name == PodInfraContainerName && utsMode != "host" {
		err = addNDotsOption(containerInfo.ResolvConfPath)
	}

	return id, err
}

func addNDotsOption(resolvFilePath string) error {
	if len(resolvFilePath) == 0 {
		glog.Errorf("DNS ResolvConfPath is empty.")
		return nil
	}

	if _, err := os.Stat(resolvFilePath); os.IsNotExist(err) {
		return fmt.Errorf("DNS ResolvConfPath specified but does not exist. It could not be updated: %s", resolvFilePath)
	}

	glog.V(4).Infof("DNS ResolvConfPath exists: %s. Will attempt to add ndots option: %s", resolvFilePath, ndotsDNSOption)

	if err := appendToFile(resolvFilePath, ndotsDNSOption); err != nil {
		glog.Errorf("resolv.conf could not be updated. err:%v", err)
		return err
	}
	return nil
}

func appendToFile(filePath, stringToAppend string) error {
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(stringToAppend)
	return err
}

// createPodInfraContainer starts the pod infra container for a pod. Returns the docker container ID of the newly created container.
func (dm *DockerManager) createPodInfraContainer(pod *api.Pod) (kubetypes.DockerID, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("createPodInfraContainer").Observe(metrics.SinceInMicroseconds(start))
	}()
	// Use host networking if specified.
	netNamespace := ""
	var ports []api.ContainerPort

	if dm.networkPlugin.Name() == "cni" {
		netNamespace = "none"
	}

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork {
		netNamespace = "host"
	} else {
		// Docker only exports ports from the pod infra container.  Let's
		// collect all of the relevant ports and export them.
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}

	container := &api.Container{
		Name:            PodInfraContainerName,
		Image:           dm.podInfraContainerImage,
		Ports:           ports,
		ImagePullPolicy: podInfraContainerImagePullPolicy,
	}

	// No pod secrets for the infra container.
	// The message isnt needed for the Infra container
	if err, _ := dm.imagePuller.PullImage(pod, container, nil); err != nil {
		return "", err
	}

	id, err := dm.runContainerInPod(pod, container, netNamespace, getIPCMode(pod), getPidMode(pod))
	if err != nil {
		return "", err
	}

	return kubetypes.DockerID(id.ID), nil
}

// TODO(vmarmol): This will soon be made non-public when its only use is internal.
// Structure keeping information on changes that need to happen for a pod. The semantics is as follows:
// - startInfraContainer is true if new Infra Containers have to be started and old one (if running) killed.
//   Additionally if it is true then containersToKeep have to be empty
// - infraContainerId have to be set if and only if startInfraContainer is false. It stores dockerID of running Infra Container
// - containersToStart keeps indices of Specs of containers that have to be started.
// - containersToKeep stores mapping from dockerIDs of running containers to indices of their Specs for containers that
//   should be kept running. If startInfraContainer is false then it contains an entry for infraContainerId (mapped to -1).
//   It shouldn't be the case where containersToStart is empty and containersToKeep contains only infraContainerId. In such case
//   Infra Container should be killed, hence it's removed from this map.
// - all running containers which are NOT contained in containersToKeep should be killed.
type empty struct{}
type PodContainerChangesSpec struct {
	StartInfraContainer bool
	InfraContainerId    kubetypes.DockerID
	ContainersToStart   map[int]empty
	ContainersToKeep    map[kubetypes.DockerID]int
}

func (dm *DockerManager) computePodContainerChanges(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus) (PodContainerChangesSpec, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("computePodContainerChanges").Observe(metrics.SinceInMicroseconds(start))
	}()

	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID
	glog.V(4).Infof("Syncing Pod %+v, podFullName: %q, uid: %q", pod, podFullName, uid)

	containersToStart := make(map[int]empty)
	containersToKeep := make(map[kubetypes.DockerID]int)

	var err error
	var podInfraContainerID kubetypes.DockerID
	var changed bool
	podInfraContainer := runningPod.FindContainerByName(PodInfraContainerName)
	if podInfraContainer != nil {
		glog.V(4).Infof("Found pod infra container for %q", podFullName)
		changed, err = dm.podInfraContainerChanged(pod, podInfraContainer)
		if err != nil {
			return PodContainerChangesSpec{}, err
		}
	}

	createPodInfraContainer := true
	if podInfraContainer == nil {
		glog.V(2).Infof("Need to restart pod infra container for %q because it is not found", podFullName)
	} else if changed {
		glog.V(2).Infof("Need to restart pod infra container for %q because it is changed", podFullName)
	} else {
		glog.V(4).Infof("Pod infra container looks good, keep it %q", podFullName)
		createPodInfraContainer = false
		podInfraContainerID = kubetypes.DockerID(podInfraContainer.ID.ID)
		containersToKeep[podInfraContainerID] = -1
	}

	for index, container := range pod.Spec.Containers {
		expectedHash := kubecontainer.HashContainer(&container)

		c := runningPod.FindContainerByName(container.Name)
		if c == nil {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, &podStatus) {
				// If we are here it means that the container is dead and should be restarted, or never existed and should
				// be created. We may be inserting this ID again if the container has changed and it has
				// RestartPolicy::Always, but it's not a big deal.
				glog.V(3).Infof("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				containersToStart[index] = empty{}
			}
			continue
		}

		containerID := kubetypes.DockerID(c.ID.ID)
		hash := c.Hash
		glog.V(3).Infof("pod %q container %q exists as %v", podFullName, container.Name, containerID)

		if createPodInfraContainer {
			// createPodInfraContainer == true and Container exists
			// If we're creating infra container everything will be killed anyway
			// If RestartPolicy is Always or OnFailure we restart containers that were running before we
			// killed them when restarting Infra Container.
			if pod.Spec.RestartPolicy != api.RestartPolicyNever {
				glog.V(1).Infof("Infra Container is being recreated. %q will be restarted.", container.Name)
				containersToStart[index] = empty{}
			}
			continue
		}

		// At this point, the container is running and pod infra container is good.
		// We will look for changes and check healthiness for the container.
		containerChanged := hash != 0 && hash != expectedHash
		if containerChanged {
			glog.Infof("pod %q container %q hash changed (%d vs %d), it will be killed and re-created.", podFullName, container.Name, hash, expectedHash)
			containersToStart[index] = empty{}
			continue
		}

		result, err := dm.prober.ProbeLiveness(pod, podStatus, container, c.ID, c.Created)
		if err != nil {
			// TODO(vmarmol): examine this logic.
			glog.V(2).Infof("probe no-error: %q", container.Name)
			containersToKeep[containerID] = index
			continue
		}
		if result == probe.Success {
			glog.V(4).Infof("probe success: %q", container.Name)
			containersToKeep[containerID] = index
			continue
		}
		if pod.Spec.RestartPolicy != api.RestartPolicyNever {
			glog.Infof("pod %q container %q is unhealthy (probe result: %v), it will be killed and re-created.", podFullName, container.Name, result)
			containersToStart[index] = empty{}
		}
	}

	// After the loop one of the following should be true:
	// - createPodInfraContainer is true and containersToKeep is empty.
	// (In fact, when createPodInfraContainer is false, containersToKeep will not be touched).
	// - createPodInfraContainer is false and containersToKeep contains at least ID of Infra Container

	// If Infra container is the last running one, we don't want to keep it.
	if !createPodInfraContainer && len(containersToStart) == 0 && len(containersToKeep) == 1 {
		containersToKeep = make(map[kubetypes.DockerID]int)
	}

	return PodContainerChangesSpec{
		StartInfraContainer: createPodInfraContainer,
		InfraContainerId:    podInfraContainerID,
		ContainersToStart:   containersToStart,
		ContainersToKeep:    containersToKeep,
	}, nil
}

// updateReasonCache updates the failure reason based on the latest error.
func (dm *DockerManager) updateReasonCache(pod *api.Pod, container *api.Container, briefError string, err error) {
	if briefError == "" || err == nil {
		return
	}
	errString := err.Error()
	dm.reasonCache.Add(pod.UID, container.Name, briefError, errString)
}

// clearReasonCache removes the entry in the reason cache.
func (dm *DockerManager) clearReasonCache(pod *api.Pod, container *api.Container) {
	dm.reasonCache.Remove(pod.UID, container.Name)
}

// Sync the running pod to match the specified desired pod.
func (dm *DockerManager) SyncPod(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus, pullSecrets []api.Secret, backOff *util.Backoff) error {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("SyncPod").Observe(metrics.SinceInMicroseconds(start))
	}()

	podFullName := kubecontainer.GetPodFullName(pod)
	containerChanges, err := dm.computePodContainerChanges(pod, runningPod, podStatus)
	if err != nil {
		return err
	}
	glog.V(3).Infof("Got container changes for pod %q: %+v", podFullName, containerChanges)

	if containerChanges.StartInfraContainer || (len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0) {
		if len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0 {
			glog.V(4).Infof("Killing Infra Container for %q because all other containers are dead.", podFullName)
		} else {
			glog.V(4).Infof("Killing Infra Container for %q, will start new one", podFullName)
		}

		// Killing phase: if we want to start new infra container, or nothing is running kill everything (including infra container)
		err = dm.KillPod(pod, runningPod)
		if err != nil {
			return err
		}
	} else {
		// Otherwise kill any containers in this pod which are not specified as ones to keep.
		for _, container := range runningPod.Containers {
			_, keep := containerChanges.ContainersToKeep[kubetypes.DockerID(container.ID.ID)]
			if !keep {
				glog.V(3).Infof("Killing unwanted container %+v", container)
				// attempt to find the appropriate container policy
				var podContainer *api.Container
				for i, c := range pod.Spec.Containers {
					if c.Name == container.Name {
						podContainer = &pod.Spec.Containers[i]
						break
					}
				}
				err = dm.KillContainerInPod(container.ID, podContainer, pod)
				if err != nil {
					glog.Errorf("Error killing container: %v", err)
				}
			}
		}
	}

	// If we should create infra container then we do it first.
	podInfraContainerID := containerChanges.InfraContainerId
	if containerChanges.StartInfraContainer && (len(containerChanges.ContainersToStart) > 0) {
		glog.V(4).Infof("Creating pod infra container for %q", podFullName)
		podInfraContainerID, err = dm.createPodInfraContainer(pod)

		// Call the networking plugin
		if err == nil {
			err = dm.networkPlugin.SetUpPod(pod.Namespace, pod.Name, podInfraContainerID)
		}
		if err != nil {
			glog.Errorf("Failed to create pod infra container: %v; Skipping pod %q", err, podFullName)
			return err
		}

		// Setup the host interface (FIXME: move to networkPlugin when ready)
		podInfraContainer, err := dm.client.InspectContainer(string(podInfraContainerID))
		if err != nil {
			glog.Errorf("Failed to inspect pod infra container: %v; Skipping pod %q", err, podFullName)
			return err
		}
		if err = hairpin.SetUpContainer(podInfraContainer.State.Pid, "eth0"); err != nil {
			glog.Warningf("Hairpin setup failed for pod %q: %v", podFullName, err)
		}
	}

	// Start everything
	for idx := range containerChanges.ContainersToStart {
		container := &pod.Spec.Containers[idx]

		// containerChanges.StartInfraContainer causes the containers to be restarted for config reasons
		// ignore backoff
		if !containerChanges.StartInfraContainer && dm.doBackOff(pod, container, podStatus, backOff) {
			glog.V(4).Infof("Backing Off restarting container %+v in pod %v", container, podFullName)
			continue
		}
		glog.V(4).Infof("Creating container %+v in pod %v", container, podFullName)
		err, msg := dm.imagePuller.PullImage(pod, container, pullSecrets)
		if err != nil {
			dm.updateReasonCache(pod, container, err.Error(), errors.New(msg))
			continue
		}

		if container.SecurityContext != nil && container.SecurityContext.RunAsNonRoot {
			err := dm.verifyNonRoot(container)
			dm.updateReasonCache(pod, container, "VerifyNonRootError", err)
			if err != nil {
				glog.Errorf("Error running pod %q container %q: %v", kubecontainer.GetPodFullName(pod), container.Name, err)
				continue
			}
		}

		// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
		// Note: when configuring the pod's containers anything that can be configured by pointing
		// to the namespace of the infra container should use namespaceMode.  This includes things like the net namespace
		// and IPC namespace.  PID mode cannot point to another container right now.
		// See createPodInfraContainer for infra container setup.
		namespaceMode := fmt.Sprintf("container:%v", podInfraContainerID)
		_, err = dm.runContainerInPod(pod, container, namespaceMode, namespaceMode, getPidMode(pod))
		dm.updateReasonCache(pod, container, kubecontainer.ErrRunContainer.Error(), err)
		if err != nil {
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running pod %q container %q: %v", kubecontainer.GetPodFullName(pod), container.Name, err)
			continue
		}
		// Successfully started the container; clear the entry in the failure
		// reason cache.
		dm.clearReasonCache(pod, container)
	}

	return nil
}

// verifyNonRoot returns an error if the container or image will run as the root user.
func (dm *DockerManager) verifyNonRoot(container *api.Container) error {
	if securitycontext.HasRunAsUser(container) {
		if securitycontext.HasRootRunAsUser(container) {
			return fmt.Errorf("container's runAsUser breaks non-root policy")
		}
		return nil
	}

	imgRoot, err := dm.isImageRoot(container.Image)
	if err != nil {
		return err
	}
	if imgRoot {
		return fmt.Errorf("container has no runAsUser and image will run as root")
	}

	return nil
}

// isImageRoot returns true if the user directive is not set on the image, the user is set to 0
// or the user is set to root.  If there is an error inspecting the image this method will return
// false and return the error.
func (dm *DockerManager) isImageRoot(image string) (bool, error) {
	img, err := dm.client.InspectImage(image)
	if err != nil {
		return false, err
	}
	if img == nil || img.Config == nil {
		return false, fmt.Errorf("unable to inspect image %s, nil Config", image)
	}

	user := getUidFromUser(img.Config.User)
	// if no user is defined container will run as root
	if user == "" {
		return true, nil
	}
	// do not allow non-numeric user directives
	uid, err := strconv.Atoi(user)
	if err != nil {
		return false, fmt.Errorf("unable to validate image is non-root, non-numeric user (%s) is not allowed", user)
	}
	// user is numeric, check for 0
	return uid == 0, nil
}

// getUidFromUser splits the uid out of a uid:gid string.
func getUidFromUser(id string) string {
	if id == "" {
		return id
	}
	// split instances where the id may contain uid:gid
	if strings.Contains(id, ":") {
		return strings.Split(id, ":")[0]
	}
	// no gid, just return the id
	return id
}

func (dm *DockerManager) doBackOff(pod *api.Pod, container *api.Container, podStatus api.PodStatus, backOff *util.Backoff) bool {
	var ts unversioned.Time
	for _, containerStatus := range podStatus.ContainerStatuses {
		if containerStatus.Name != container.Name {
			continue
		}
		// first failure
		if containerStatus.State.Terminated != nil {
			ts = containerStatus.State.Terminated.FinishedAt
			break
		}
		// state is waiting and the failure timestamp is in LastTerminationState
		if (containerStatus.State.Waiting != nil) && (containerStatus.LastTerminationState.Terminated != nil) {
			ts = containerStatus.LastTerminationState.Terminated.FinishedAt
			break
		}
	}

	// found a container that requires backoff
	if !ts.IsZero() {
		dockerName := KubeletContainerName{
			PodFullName:   kubecontainer.GetPodFullName(pod),
			PodUID:        pod.UID,
			ContainerName: container.Name,
		}
		stableName, _ := BuildDockerName(dockerName, container)
		if backOff.IsInBackOffSince(stableName, ts.Time) {
			if ref, err := kubecontainer.GenerateContainerRef(pod, container); err == nil {
				dm.recorder.Eventf(ref, "Backoff", "Back-off restarting failed docker container")
			}
			err := fmt.Errorf("Back-off %s restarting failed container=%s pod=%s", backOff.Get(stableName), container.Name, kubecontainer.GetPodFullName(pod))
			dm.updateReasonCache(pod, container, kubecontainer.ErrCrashLoopBackOff.Error(), err)
			glog.Infof("%s", err.Error())
			return true
		}
		backOff.Next(stableName, ts.Time)
	}
	dm.clearReasonCache(pod, container)
	return false
}

// getPidMode returns the pid mode to use on the docker container based on pod.Spec.HostPID.
func getPidMode(pod *api.Pod) string {
	pidMode := ""
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostPID {
		pidMode = "host"
	}
	return pidMode
}

// getIPCMode returns the ipc mode to use on the docker container based on pod.Spec.HostIPC.
func getIPCMode(pod *api.Pod) string {
	ipcMode := ""
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostIPC {
		ipcMode = "host"
	}
	return ipcMode
}

// GetNetNs returns the network namespace path for the given container
func (dm *DockerManager) GetNetNs(containerID kubecontainer.ContainerID) (string, error) {
	inspectResult, err := dm.client.InspectContainer(containerID.ID)
	if err != nil {
		glog.Errorf("Error inspecting container: '%v'", err)
		return "", err
	}
	netnsPath := fmt.Sprintf(DockerNetnsFmt, inspectResult.State.Pid)
	return netnsPath, nil
}

// Garbage collection of dead containers
func (dm *DockerManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy) error {
	return dm.containerGC.GarbageCollect(gcPolicy)
}
