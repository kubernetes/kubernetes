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
	"strconv"
	"strings"
	"sync"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerstrslice "github.com/docker/engine-api/types/strslice"
	dockerversion "github.com/docker/engine-api/types/versions"
	dockernat "github.com/docker/go-connections/nat"
	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/hairpin"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
)

const (
	DockerType = "docker"

	minimumDockerAPIVersion = "1.20"

	dockerv110APIVersion = "1.21"

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

	DockerNetnsFmt = "/proc/%v/ns/net"

	// String used to detect docker host mode for various namespaces (e.g.
	// networking). Must match the value returned by docker inspect -f
	// '{{.HostConfig.NetworkMode}}'.
	namespaceModeHost = "host"

	// Remote API version for docker daemon version v1.10
	// https://docs.docker.com/engine/reference/api/docker_remote_api/
	dockerV110APIVersion = "1.22"

	// The expiration time of version cache.
	versionCacheTTL = 60 * time.Second
)

var (
	// DockerManager implements the Runtime interface.
	_ kubecontainer.Runtime = &DockerManager{}

	// TODO: make this a TTL based pull (if image older than X policy, pull)
	podInfraContainerImagePullPolicy = api.PullIfNotPresent

	// Default set of security options. Seccomp is disabled by default until
	// github issue #20870 is resolved.
	defaultSecurityOpt = []string{"seccomp:unconfined"}
)

type DockerManager struct {
	client              DockerInterface
	recorder            record.EventRecorder
	containerRefManager *kubecontainer.RefManager
	os                  kubecontainer.OSInterface
	machineInfo         *cadvisorapi.MachineInfo

	// The image name of the pod infra container.
	podInfraContainerImage string
	// (Optional) Additional environment variables to be set for the pod infra container.
	podInfraContainerEnv []api.EnvVar

	// TODO(yifan): Record the pull failure so we can eliminate the image checking?
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

	// Health check results.
	livenessManager proberesults.Manager

	// RuntimeHelper that wraps kubelet to generate runtime container options.
	runtimeHelper kubecontainer.RuntimeHelper

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner

	// Handler used to execute commands in containers.
	execHandler ExecHandler

	// Used to set OOM scores of processes.
	oomAdjuster *oom.OOMAdjuster

	// Get information from /proc mount.
	procFs procfs.ProcFSInterface

	// If true, enforce container cpu limits with CFS quota support
	cpuCFSQuota bool

	// Container GC manager
	containerGC *containerGC

	// Support for gathering custom metrics.
	enableCustomMetrics bool

	// If true, the "hairpin mode" flag is set on container interfaces.
	// A false value means the kubelet just backs off from setting it,
	// it might already be true.
	configureHairpinMode bool

	// Provides image stats
	*imageStatsProvider

	// The version cache of docker daemon.
	versionCache *cache.ObjectCache
}

// A subset of the pod.Manager interface extracted for testing purposes.
type podGetter interface {
	GetPodByUID(types.UID) (*api.Pod, bool)
}

func PodInfraContainerEnv(env map[string]string) kubecontainer.Option {
	return func(rt kubecontainer.Runtime) {
		dm := rt.(*DockerManager)
		for k, v := range env {
			dm.podInfraContainerEnv = append(dm.podInfraContainerEnv, api.EnvVar{
				Name:  k,
				Value: v,
			})
		}
	}
}

func NewDockerManager(
	client DockerInterface,
	recorder record.EventRecorder,
	livenessManager proberesults.Manager,
	containerRefManager *kubecontainer.RefManager,
	podGetter podGetter,
	machineInfo *cadvisorapi.MachineInfo,
	podInfraContainerImage string,
	qps float32,
	burst int,
	containerLogsDir string,
	osInterface kubecontainer.OSInterface,
	networkPlugin network.NetworkPlugin,
	runtimeHelper kubecontainer.RuntimeHelper,
	httpClient kubetypes.HttpGetter,
	execHandler ExecHandler,
	oomAdjuster *oom.OOMAdjuster,
	procFs procfs.ProcFSInterface,
	cpuCFSQuota bool,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
	enableCustomMetrics bool,
	hairpinMode bool,
	options ...kubecontainer.Option) *DockerManager {
	// Wrap the docker client with instrumentedDockerInterface
	client = newInstrumentedDockerInterface(client)

	// Work out the location of the Docker runtime, defaulting to /var/lib/docker
	// if there are any problems.
	dockerRoot := "/var/lib/docker"
	dockerInfo, err := client.Info()
	if err != nil {
		glog.Errorf("Failed to execute Info() call to the Docker client: %v", err)
		glog.Warningf("Using fallback default of /var/lib/docker for location of Docker runtime")
	} else {
		dockerRoot = dockerInfo.DockerRootDir
		glog.Infof("Setting dockerRoot to %s", dockerRoot)
	}

	dm := &DockerManager{
		client:                 client,
		recorder:               recorder,
		containerRefManager:    containerRefManager,
		os:                     osInterface,
		machineInfo:            machineInfo,
		podInfraContainerImage: podInfraContainerImage,
		dockerPuller:           newDockerPuller(client, qps, burst),
		dockerRoot:             dockerRoot,
		containerLogsDir:       containerLogsDir,
		networkPlugin:          networkPlugin,
		livenessManager:        livenessManager,
		runtimeHelper:          runtimeHelper,
		execHandler:            execHandler,
		oomAdjuster:            oomAdjuster,
		procFs:                 procFs,
		cpuCFSQuota:            cpuCFSQuota,
		enableCustomMetrics:    enableCustomMetrics,
		configureHairpinMode:   hairpinMode,
		imageStatsProvider:     &imageStatsProvider{client},
	}
	dm.runner = lifecycle.NewHandlerRunner(httpClient, dm, dm)
	if serializeImagePulls {
		dm.imagePuller = kubecontainer.NewSerializedImagePuller(kubecontainer.FilterEventRecorder(recorder), dm, imageBackOff)
	} else {
		dm.imagePuller = kubecontainer.NewImagePuller(kubecontainer.FilterEventRecorder(recorder), dm, imageBackOff)
	}
	dm.containerGC = NewContainerGC(client, podGetter, containerLogsDir)

	dm.versionCache = cache.NewObjectCache(
		func() (interface{}, error) {
			return dm.getVersionInfo()
		},
		versionCacheTTL,
	)

	// apply optional settings..
	for _, optf := range options {
		optf(dm)
	}

	return dm
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
	opts := dockertypes.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Since:      strconv.FormatInt(since, 10),
		Timestamps: logOptions.Timestamps,
		Follow:     logOptions.Follow,
	}
	if logOptions.TailLines != nil {
		opts.Tail = strconv.FormatInt(*logOptions.TailLines, 10)
	}
	sopts := StreamOptions{
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  false,
	}
	err = dm.client.Logs(containerID.ID, opts, sopts)
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

// determineContainerIP determines the IP address of the given container.  It is expected
// that the container passed is the infrastructure container of a pod and the responsibility
// of the caller to ensure that the correct container is passed.
func (dm *DockerManager) determineContainerIP(podNamespace, podName string, container *dockertypes.ContainerJSON) string {
	result := ""

	if container.NetworkSettings != nil {
		result = container.NetworkSettings.IPAddress
	}

	if dm.networkPlugin.Name() != network.DefaultPluginName {
		netStatus, err := dm.networkPlugin.Status(podNamespace, podName, kubecontainer.DockerID(container.ID).ContainerID())
		if err != nil {
			glog.Errorf("NetworkPlugin %s failed on the status hook for pod '%s' - %v", dm.networkPlugin.Name(), podName, err)
		} else if netStatus != nil {
			result = netStatus.IP.String()
		}
	}

	return result
}

func (dm *DockerManager) inspectContainer(id string, podName, podNamespace string) (*kubecontainer.ContainerStatus, string, error) {
	var ip string
	iResult, err := dm.client.InspectContainer(id)
	if err != nil {
		return nil, ip, err
	}
	glog.V(4).Infof("Container inspect result: %+v", *iResult)

	// TODO: Get k8s container name by parsing the docker name. This will be
	// replaced by checking docker labels eventually.
	dockerName, hash, err := ParseDockerName(iResult.Name)
	if err != nil {
		return nil, ip, fmt.Errorf("Unable to parse docker name %q", iResult.Name)
	}
	containerName := dockerName.ContainerName

	var containerInfo *labelledContainerInfo
	containerInfo = getContainerInfoFromLabel(iResult.Config.Labels)

	parseTimestampError := func(label, s string) {
		glog.Errorf("Failed to parse %q timestamp %q for container %q of pod %q", label, s, id, kubecontainer.BuildPodFullName(podName, podNamespace))
	}
	var createdAt, startedAt, finishedAt time.Time
	if createdAt, err = parseDockerTimestamp(iResult.Created); err != nil {
		parseTimestampError("Created", iResult.Created)
	}
	if startedAt, err = parseDockerTimestamp(iResult.State.StartedAt); err != nil {
		parseTimestampError("StartedAt", iResult.State.StartedAt)
	}
	if finishedAt, err = parseDockerTimestamp(iResult.State.FinishedAt); err != nil {
		parseTimestampError("FinishedAt", iResult.State.FinishedAt)
	}

	status := kubecontainer.ContainerStatus{
		Name:         containerName,
		RestartCount: containerInfo.RestartCount,
		Image:        iResult.Config.Image,
		ImageID:      DockerPrefix + iResult.Image,
		ID:           kubecontainer.DockerID(id).ContainerID(),
		ExitCode:     iResult.State.ExitCode,
		CreatedAt:    createdAt,
		Hash:         hash,
	}
	if iResult.State.Running {
		// Container that are running, restarting and paused
		status.State = kubecontainer.ContainerStateRunning
		status.StartedAt = startedAt
		if containerName == PodInfraContainerName {
			ip = dm.determineContainerIP(podNamespace, podName, iResult)
		}
		return &status, ip, nil
	}

	// Find containers that have exited or failed to start.
	if !finishedAt.IsZero() || iResult.State.ExitCode != 0 {
		// Containers that are exited, dead or created (docker failed to start container)
		// When a container fails to start State.ExitCode is non-zero, FinishedAt and StartedAt are both zero
		reason := ""
		message := iResult.State.Error

		// Note: An application might handle OOMKilled gracefully.
		// In that case, the container is oom killed, but the exit
		// code could be 0.
		if iResult.State.OOMKilled {
			reason = "OOMKilled"
		} else if iResult.State.ExitCode == 0 {
			reason = "Completed"
		} else if !finishedAt.IsZero() {
			reason = "Error"
		} else {
			// finishedAt is zero and ExitCode is nonZero occurs when docker fails to start the container
			reason = ErrContainerCannotRun.Error()
			// Adjust time to the time docker attempted to run the container, otherwise startedAt and finishedAt will be set to epoch, which is misleading
			finishedAt = createdAt
			startedAt = createdAt
		}

		terminationMessagePath := containerInfo.TerminationMessagePath
		if terminationMessagePath != "" {
			for _, mount := range iResult.Mounts {
				if mount.Destination == terminationMessagePath {
					path := mount.Source
					if data, err := ioutil.ReadFile(path); err != nil {
						message = fmt.Sprintf("Error on reading termination-log %s: %v", path, err)
					} else {
						message = string(data)
					}
				}
			}
		}
		status.State = kubecontainer.ContainerStateExited
		status.Message = message
		status.Reason = reason
		status.StartedAt = startedAt
		status.FinishedAt = finishedAt
	} else {
		// Non-running containers that are created (not yet started or kubelet failed before calling
		// start container function etc.) Kubelet doesn't handle these scenarios yet.
		status.State = kubecontainer.ContainerStateUnknown
	}
	return &status, "", nil
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
// '<HostPath>:<ContainerPath>:ro', if the path is read only, or
// '<HostPath>:<ContainerPath>:Z', if the volume requires SELinux
// relabeling and the pod provides an SELinux label
func makeMountBindings(mounts []kubecontainer.Mount, podHasSELinuxLabel bool) (result []string) {
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.HostPath, m.ContainerPath)
		if m.ReadOnly {
			bind += ":ro"
		}
		// Only request relabeling if the pod provides an
		// SELinux context. If the pod does not provide an
		// SELinux context relabeling will label the volume
		// with the container's randomly allocated MCS label.
		// This would restrict access to the volume to the
		// container which mounts it first.
		if m.SELinuxRelabel && podHasSELinuxLabel {
			if m.ReadOnly {
				bind += ",Z"
			} else {
				bind += ":Z"
			}

		}
		result = append(result, bind)
	}
	return
}

func makePortsAndBindings(portMappings []kubecontainer.PortMapping) (map[dockernat.Port]struct{}, map[dockernat.Port][]dockernat.PortBinding) {
	exposedPorts := map[dockernat.Port]struct{}{}
	portBindings := map[dockernat.Port][]dockernat.PortBinding{}
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

		dockerPort := dockernat.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}

		hostBinding := dockernat.PortBinding{
			HostPort: strconv.Itoa(exteriorPort),
			HostIP:   port.HostIP,
		}

		// Allow multiple host ports bind to same docker port
		if existedBindings, ok := portBindings[dockerPort]; ok {
			// If a docker port already map to a host port, just append the host ports
			portBindings[dockerPort] = append(existedBindings, hostBinding)
		} else {
			// Otherwise, it's fresh new port binding
			portBindings[dockerPort] = []dockernat.PortBinding{
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
	pidMode string,
	restartCount int,
	oomScoreAdj int) (kubecontainer.ContainerID, error) {

	dockerName := KubeletContainerName{
		PodFullName:   kubecontainer.GetPodFullName(pod),
		PodUID:        pod.UID,
		ContainerName: container.Name,
	}

	securityOpts, err := dm.defaultSecurityOpt()
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}

	// Pod information is recorded on the container as labels to preserve it in the event the pod is deleted
	// while the Kubelet is down and there is no information available to recover the pod.
	// TODO: keep these labels up to date if the pod changes
	labels := newLabels(container, pod, restartCount, dm.enableCustomMetrics)

	// TODO(random-liu): Remove this when we start to use new labels for KillContainerInPod
	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// TODO: This is kind of hacky, we should really just encode the bits we need.
		// TODO: This is hacky because the Kubelet should be parameterized to encode a specific version
		//   and needs to be able to migrate this whenever we deprecate v1. Should be a member of DockerManager.
		if data, err := runtime.Encode(api.Codecs.LegacyCodec(unversioned.GroupVersion{Group: api.GroupName, Version: "v1"}), pod); err == nil {
			labels[kubernetesPodLabel] = string(data)
		} else {
			glog.Errorf("Failed to encode pod: %s for prestop hook", pod.Name)
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
	podHasSELinuxLabel := pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxOptions != nil
	binds := makeMountBindings(opts.Mounts, podHasSELinuxLabel)
	// The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
	// TODO(yifan): Consider to pull this logic out since we might need to reuse it in
	// other container runtime.
	_, containerName, cid := BuildDockerName(dockerName, container)
	if opts.PodContainerDir != "" && len(container.TerminationMessagePath) != 0 {
		// Because the PodContainerDir contains pod uid and container name which is unique enough,
		// here we just add an unique container id to make the path unique for different instances
		// of the same container.
		containerLogPath := path.Join(opts.PodContainerDir, cid)
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

	hc := &dockercontainer.HostConfig{
		Binds:          binds,
		NetworkMode:    dockercontainer.NetworkMode(netMode),
		IpcMode:        dockercontainer.IpcMode(ipcMode),
		UTSMode:        dockercontainer.UTSMode(utsMode),
		PidMode:        dockercontainer.PidMode(pidMode),
		ReadonlyRootfs: readOnlyRootFilesystem(container),
		Resources: dockercontainer.Resources{
			Memory:     memoryLimit,
			MemorySwap: -1,
			CPUShares:  cpuShares,
		},
		SecurityOpt: securityOpts,
	}

	// If current api version is newer than docker 1.10 requested, set OomScoreAdj to HostConfig
	result, err := dm.checkDockerAPIVersion(dockerv110APIVersion)
	if err != nil {
		glog.Errorf("Failed to check docker api version: %v", err)
	} else if result >= 0 {
		hc.OomScoreAdj = oomScoreAdj
	}

	if dm.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned to allow full usage of cpu resource.
		cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())

		hc.CPUQuota = cpuQuota
		hc.CPUPeriod = cpuPeriod
	}

	if len(opts.CgroupParent) > 0 {
		hc.CgroupParent = opts.CgroupParent
	}

	dockerOpts := dockertypes.ContainerCreateConfig{
		Name: containerName,
		Config: &dockercontainer.Config{
			Env:        makeEnvList(opts.Envs),
			Image:      container.Image,
			WorkingDir: container.WorkingDir,
			Labels:     labels,
			// Interactive containers:
			OpenStdin: container.Stdin,
			StdinOnce: container.StdinOnce,
			Tty:       container.TTY,
		},
		HostConfig: hc,
	}

	// Set network configuration for infra-container
	if container.Name == PodInfraContainerName {
		setInfraContainerNetworkConfig(pod, netMode, opts, dockerOpts)
	}

	setEntrypointAndCommand(container, opts, dockerOpts)

	glog.V(3).Infof("Container %v/%v/%v: setting entrypoint \"%v\" and command \"%v\"", pod.Namespace, pod.Name, container.Name, dockerOpts.Config.Entrypoint, dockerOpts.Config.Cmd)

	securityContextProvider := securitycontext.NewSimpleSecurityContextProvider()
	securityContextProvider.ModifyContainerConfig(pod, container, dockerOpts.Config)
	securityContextProvider.ModifyHostConfig(pod, container, dockerOpts.HostConfig)
	createResp, err := dm.client.CreateContainer(dockerOpts)
	if err != nil {
		dm.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.FailedToCreateContainer, "Failed to create docker container with error: %v", err)
		return kubecontainer.ContainerID{}, err
	}
	if len(createResp.Warnings) != 0 {
		glog.V(2).Infof("Container %q of pod %q created with warnings: %v", container.Name, format.Pod(pod), createResp.Warnings)
	}
	dm.recorder.Eventf(ref, api.EventTypeNormal, kubecontainer.CreatedContainer, "Created container with docker id %v", utilstrings.ShortenString(createResp.ID, 12))

	if err = dm.client.StartContainer(createResp.ID); err != nil {
		dm.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.FailedToStartContainer,
			"Failed to start container with docker id %v with error: %v", utilstrings.ShortenString(createResp.ID, 12), err)
		return kubecontainer.ContainerID{}, err
	}
	dm.recorder.Eventf(ref, api.EventTypeNormal, kubecontainer.StartedContainer, "Started container with docker id %v", utilstrings.ShortenString(createResp.ID, 12))

	return kubecontainer.DockerID(createResp.ID).ContainerID(), nil
}

// setInfraContainerNetworkConfig sets the network configuration for the infra-container. We only set network configuration for infra-container, all
// the user containers will share the same network namespace with infra-container.
func setInfraContainerNetworkConfig(pod *api.Pod, netMode string, opts *kubecontainer.RunContainerOptions, dockerOpts dockertypes.ContainerCreateConfig) {
	exposedPorts, portBindings := makePortsAndBindings(opts.PortMappings)
	dockerOpts.Config.ExposedPorts = exposedPorts
	dockerOpts.HostConfig.PortBindings = dockernat.PortMap(portBindings)

	if netMode != namespaceModeHost {
		dockerOpts.Config.Hostname = opts.Hostname
		if len(opts.DNS) > 0 {
			dockerOpts.HostConfig.DNS = opts.DNS
		}
		if len(opts.DNSSearch) > 0 {
			dockerOpts.HostConfig.DNSSearch = opts.DNSSearch
		}
	}
}

func setEntrypointAndCommand(container *api.Container, opts *kubecontainer.RunContainerOptions, dockerOpts dockertypes.ContainerCreateConfig) {
	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)

	dockerOpts.Config.Entrypoint = dockerstrslice.StrSlice(command)
	dockerOpts.Config.Cmd = dockerstrslice.StrSlice(args)
}

// A helper function to get the KubeletContainerName and hash from a docker
// container.
func getDockerContainerNameInfo(c *dockertypes.Container) (*KubeletContainerName, uint64, error) {
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
func getPodInfoFromContainer(c *dockertypes.Container) (types.UID, string, string, error) {
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
	for _, p := range pods {
		result = append(result, p)
	}
	return result, nil
}

// List all images in the local storage.
func (dm *DockerManager) ListImages() ([]kubecontainer.Image, error) {
	var images []kubecontainer.Image

	dockerImages, err := dm.client.ListImages(dockertypes.ImageListOptions{})
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
	// TODO(harryz) currently Runtime interface does not provide other remove options.
	_, err := dm.client.RemoveImage(image.Image, dockertypes.ImageRemoveOptions{})
	return err
}

// podInfraContainerChanged returns true if the pod infra container has changed.
func (dm *DockerManager) podInfraContainerChanged(pod *api.Pod, podInfraContainerStatus *kubecontainer.ContainerStatus) (bool, error) {
	var ports []api.ContainerPort

	// Check network mode.
	if kubecontainer.IsHostNetworkPod(pod) {
		dockerPodInfraContainer, err := dm.client.InspectContainer(podInfraContainerStatus.ID.ID)
		if err != nil {
			return false, err
		}

		networkMode := getDockerNetworkMode(dockerPodInfraContainer)
		if networkMode != namespaceModeHost {
			glog.V(4).Infof("host: %v, %v", pod.Spec.SecurityContext.HostNetwork, networkMode)
			return true, nil
		}
	} else if dm.networkPlugin.Name() != "cni" && dm.networkPlugin.Name() != "kubenet" {
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
		Env:             dm.podInfraContainerEnv,
	}
	return podInfraContainerStatus.Hash != kubecontainer.HashContainer(expectedPodInfraContainer), nil
}

// determine if the container root should be a read only filesystem.
func readOnlyRootFilesystem(container *api.Container) bool {
	return container.SecurityContext != nil && container.SecurityContext.ReadOnlyRootFilesystem != nil && *container.SecurityContext.ReadOnlyRootFilesystem
}

// container must not be nil
func getDockerNetworkMode(container *dockertypes.ContainerJSON) string {
	if container.HostConfig != nil {
		return string(container.HostConfig.NetworkMode)
	}
	return ""
}

// dockerVersion implementes kubecontainer.Version interface by implementing
// Compare() and String(). It could contain either server version or api version.
type dockerVersion string

func (v dockerVersion) String() string {
	return string(v)
}

func (v dockerVersion) Compare(other string) (int, error) {
	if dockerversion.LessThan(string(v), other) {
		return -1, nil
	} else if dockerversion.GreaterThan(string(v), other) {
		return 1, nil
	}
	return 0, nil
}

func (dm *DockerManager) Type() string {
	return DockerType
}

func (dm *DockerManager) Version() (kubecontainer.Version, error) {
	v, err := dm.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}

	return dockerVersion(v.Version), nil
}

func (dm *DockerManager) APIVersion() (kubecontainer.Version, error) {
	v, err := dm.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}

	return dockerVersion(v.APIVersion), nil
}

// Status returns error if docker daemon is unhealthy, nil otherwise.
// Now we do this by checking whether:
// 1) `docker version` works
// 2) docker version is compatible with minimum requirement
func (dm *DockerManager) Status() error {
	return dm.checkVersionCompatibility()
}

func (dm *DockerManager) checkVersionCompatibility() error {
	version, err := dm.APIVersion()
	if err != nil {
		return err
	}
	// Verify the docker version.
	result, err := version.Compare(minimumDockerAPIVersion)
	if err != nil {
		return fmt.Errorf("failed to compare current docker version %v with minimum support Docker version %q - %v", version, minimumDockerAPIVersion, err)
	}
	if result < 0 {
		return fmt.Errorf("container runtime version is older than %s", minimumDockerAPIVersion)
	}
	return nil
}

func (dm *DockerManager) defaultSecurityOpt() ([]string, error) {
	version, err := dm.APIVersion()
	if err != nil {
		return nil, err
	}
	// seccomp is to be disabled on docker versions >= v1.10
	result, err := version.Compare(dockerV110APIVersion)
	if err != nil {
		return nil, err
	}
	if result >= 0 {
		return defaultSecurityOpt, nil
	}
	return nil, nil
}

// RunInContainer run the command inside the container identified by containerID
func (dm *DockerManager) RunInContainer(containerID kubecontainer.ContainerID, cmd []string) ([]byte, error) {
	glog.V(2).Infof("Using docker native exec to run cmd %+v inside container %s", cmd, containerID)
	createOpts := dockertypes.ExecConfig{
		Cmd:          cmd,
		AttachStdin:  false,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
	}
	execObj, err := dm.client.CreateExec(containerID.ID, createOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to run in container - Exec setup failed - %v", err)
	}
	var buf bytes.Buffer
	startOpts := dockertypes.ExecStartCheck{Detach: false, Tty: false}
	streamOpts := StreamOptions{
		OutputStream: &buf,
		ErrorStream:  &buf,
		RawTerminal:  false,
	}
	err = dm.client.StartExec(execObj.ID, startOpts, streamOpts)
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
	Inspect *dockertypes.ContainerExecInspect
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
	// TODO(random-liu): Do we really use the *Logs* field here?
	opts := dockertypes.ContainerAttachOptions{
		Stream: true,
		Stdin:  stdin != nil,
		Stdout: stdout != nil,
		Stderr: stderr != nil,
	}
	sopts := StreamOptions{
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  tty,
	}
	return dm.client.AttachToContainer(containerID.ID, opts, sopts)
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

	commandString := fmt.Sprintf("%s %s", nsenterPath, strings.Join(args, " "))
	glog.V(4).Infof("executing port forwarding command: %s", commandString)

	command := exec.Command(nsenterPath, args...)
	command.Stdout = stream

	stderr := new(bytes.Buffer)
	command.Stderr = stderr

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

	if err := command.Run(); err != nil {
		return fmt.Errorf("%v: %s", err, stderr.String())
	}

	return nil
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

// TODO(random-liu): Change running pod to pod status in the future. We can't do it now, because kubelet also uses this function without pod status.
// We can only deprecate this after refactoring kubelet.
// TODO(random-liu): After using pod status for KillPod(), we can also remove the kubernetesPodLabel, because all the needed information should have
// been extract from new labels and stored in pod status.
func (dm *DockerManager) KillPod(pod *api.Pod, runningPod kubecontainer.Pod) error {
	result := dm.killPodWithSyncResult(pod, runningPod)
	return result.Error()
}

// TODO(random-liu): This is just a temporary function, will be removed when we acturally add PodSyncResult
// NOTE(random-liu): The pod passed in could be *nil* when kubelet restarted.
func (dm *DockerManager) killPodWithSyncResult(pod *api.Pod, runningPod kubecontainer.Pod) (result kubecontainer.PodSyncResult) {
	// Send the kills in parallel since they may take a long time.
	// There may be len(runningPod.Containers) or len(runningPod.Containers)-1 of result in the channel
	containerResults := make(chan *kubecontainer.SyncResult, len(runningPod.Containers))
	wg := sync.WaitGroup{}
	var (
		networkContainer *kubecontainer.Container
		networkSpec      *api.Container
	)
	wg.Add(len(runningPod.Containers))
	for _, container := range runningPod.Containers {
		go func(container *kubecontainer.Container) {
			defer utilruntime.HandleCrash()
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

			killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, container.Name)
			err := dm.KillContainerInPod(container.ID, containerSpec, pod, "Need to kill pod.")
			if err != nil {
				killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
				glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, runningPod.ID)
			}
			containerResults <- killContainerResult
		}(container)
	}
	wg.Wait()
	close(containerResults)
	for containerResult := range containerResults {
		result.AddSyncResult(containerResult)
	}
	if networkContainer != nil {
		ins, err := dm.client.InspectContainer(networkContainer.ID.ID)
		if err != nil {
			err = fmt.Errorf("Error inspecting container %v: %v", networkContainer.ID.ID, err)
			glog.Error(err)
			result.Fail(err)
			return
		}
		if getDockerNetworkMode(ins) != namespaceModeHost {
			teardownNetworkResult := kubecontainer.NewSyncResult(kubecontainer.TeardownNetwork, kubecontainer.BuildPodFullName(runningPod.Name, runningPod.Namespace))
			result.AddSyncResult(teardownNetworkResult)
			glog.V(3).Infof("Calling network plugin %s to tear down pod for %s", dm.networkPlugin.Name(), kubecontainer.BuildPodFullName(runningPod.Name, runningPod.Namespace))
			if err := dm.networkPlugin.TearDownPod(runningPod.Namespace, runningPod.Name, networkContainer.ID); err != nil {
				message := fmt.Sprintf("Failed to teardown network for pod %q using network plugins %q: %v", runningPod.ID, dm.networkPlugin.Name(), err)
				teardownNetworkResult.Fail(kubecontainer.ErrTeardownNetwork, message)
				glog.Error(message)
			}
		}
		killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, networkContainer.Name)
		result.AddSyncResult(killContainerResult)
		if err := dm.KillContainerInPod(networkContainer.ID, networkSpec, pod, "Need to kill pod."); err != nil {
			killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
			glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, runningPod.ID)
		}
	}
	return
}

// KillContainerInPod kills a container in the pod. It must be passed either a container ID or a container and pod,
// and will attempt to lookup the other information if missing.
func (dm *DockerManager) KillContainerInPod(containerID kubecontainer.ContainerID, container *api.Container, pod *api.Pod, message string) error {
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
	return dm.killContainer(containerID, container, pod, message)
}

// killContainer accepts a containerID and an optional container or pod containing shutdown policies. Invoke
// KillContainerInPod if information must be retrieved first.
func (dm *DockerManager) killContainer(containerID kubecontainer.ContainerID, container *api.Container, pod *api.Pod, reason string) error {
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
			defer utilruntime.HandleCrash()
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
	err := dm.client.StopContainer(ID, int(gracePeriod))
	if err == nil {
		glog.V(2).Infof("Container %q exited after %s", name, unversioned.Now().Sub(start.Time))
	} else {
		glog.V(2).Infof("Container %q termination failed after %s: %v", name, unversioned.Now().Sub(start.Time), err)
	}
	ref, ok := dm.containerRefManager.GetRef(containerID)
	if !ok {
		glog.Warningf("No ref for pod '%q'", name)
	} else {
		message := fmt.Sprintf("Killing container with docker id %v", utilstrings.ShortenString(ID, 12))
		if reason != "" {
			message = fmt.Sprint(message, ": ", reason)
		}
		dm.recorder.Event(ref, api.EventTypeNormal, kubecontainer.KillingContainer, message)
		dm.containerRefManager.ClearRef(containerID)
	}
	return err
}

var errNoPodOnContainer = fmt.Errorf("no pod information labels on Docker container")

// containerAndPodFromLabels tries to load the appropriate container info off of a Docker container's labels
func containerAndPodFromLabels(inspect *dockertypes.ContainerJSON) (pod *api.Pod, container *api.Container, err error) {
	if inspect == nil && inspect.Config == nil && inspect.Config.Labels == nil {
		return nil, nil, errNoPodOnContainer
	}
	labels := inspect.Config.Labels

	// the pod data may not be set
	if body, found := labels[kubernetesPodLabel]; found {
		pod = &api.Pod{}
		if err = runtime.DecodeInto(api.Codecs.UniversalDecoder(), []byte(body), pod); err == nil {
			name := labels[kubernetesContainerNameLabel]
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
		if period, ok := labels[kubernetesPodTerminationGracePeriodLabel]; ok {
			if seconds, err := strconv.ParseInt(period, 10, 64); err == nil {
				pod = &api.Pod{}
				pod.DeletionGracePeriodSeconds = &seconds
			}
		}
	}

	return
}

func (dm *DockerManager) applyOOMScoreAdj(container *api.Container, containerInfo *dockertypes.ContainerJSON) error {
	cgroupName, err := dm.procFs.GetFullContainerName(containerInfo.State.Pid)
	if err != nil {
		if err == os.ErrNotExist {
			// Container exited. We cannot do anything about it. Ignore this error.
			glog.V(2).Infof("Failed to apply OOM score adj on container %q with ID %q. Init process does not exist.", containerInfo.Name, containerInfo.ID)
			return nil
		}
		return err
	}
	oomScoreAdj := dm.calculateOomScoreAdj(container)
	if err = dm.oomAdjuster.ApplyOOMScoreAdjContainer(cgroupName, oomScoreAdj, 5); err != nil {
		if err == os.ErrNotExist {
			// Container exited. We cannot do anything about it. Ignore this error.
			glog.V(2).Infof("Failed to apply OOM score adj on container %q with ID %q. Init process does not exist.", containerInfo.Name, containerInfo.ID)
			return nil
		}
		return err
	}
	return nil
}

// Run a single container from a pod. Returns the docker container ID
// If do not need to pass labels, just pass nil.
func (dm *DockerManager) runContainerInPod(pod *api.Pod, container *api.Container, netMode, ipcMode, pidMode, podIP string, restartCount int) (kubecontainer.ContainerID, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("runContainerInPod").Observe(metrics.SinceInMicroseconds(start))
	}()

	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Can't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	opts, err := dm.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP)
	if err != nil {
		return kubecontainer.ContainerID{}, fmt.Errorf("GenerateRunContainerOptions: %v", err)
	}

	utsMode := ""
	if kubecontainer.IsHostNetworkPod(pod) {
		utsMode = namespaceModeHost
	}

	oomScoreAdj := dm.calculateOomScoreAdj(container)

	id, err := dm.runContainer(pod, container, opts, ref, netMode, ipcMode, utsMode, pidMode, restartCount, oomScoreAdj)
	if err != nil {
		return kubecontainer.ContainerID{}, fmt.Errorf("runContainer: %v", err)
	}

	// Remember this reference so we can report events about this container
	if ref != nil {
		dm.containerRefManager.SetRef(id, ref)
	}

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := dm.runner.Run(id, pod, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			err := fmt.Errorf("PostStart handler: %v", handlerErr)
			dm.KillContainerInPod(id, container, pod, err.Error())
			return kubecontainer.ContainerID{}, err
		}
	}

	// Create a symbolic link to the Docker container log file using a name which captures the
	// full pod name, the container name and the Docker container ID. Cluster level logging will
	// capture these symbolic filenames which can be used for search terms in Elasticsearch or for
	// labels for Cloud Logging.
	containerLogFile := path.Join(dm.dockerRoot, "containers", id.ID, fmt.Sprintf("%s-json.log", id.ID))
	symlinkFile := LogSymlink(dm.containerLogsDir, kubecontainer.GetPodFullName(pod), container.Name, id.ID)
	if err = dm.os.Symlink(containerLogFile, symlinkFile); err != nil {
		glog.Errorf("Failed to create symbolic link to the log file of pod %q container %q: %v", format.Pod(pod), container.Name, err)
	}

	// Container information is used in adjusting OOM scores and adding ndots.
	containerInfo, err := dm.client.InspectContainer(id.ID)
	if err != nil {
		return kubecontainer.ContainerID{}, fmt.Errorf("InspectContainer: %v", err)
	}
	// Ensure the PID actually exists, else we'll move ourselves.
	if containerInfo.State.Pid == 0 {
		return kubecontainer.ContainerID{}, fmt.Errorf("can't get init PID for container %q", id)
	}

	// Check if current docker version is higher than 1.10. Otherwise, we have to apply OOMScoreAdj instead of using docker API.
	err = dm.applyOOMScoreAdjIfNeeded(container, containerInfo)
	if err != nil {
		return kubecontainer.ContainerID{}, err
	}

	// The addNDotsOption call appends the ndots option to the resolv.conf file generated by docker.
	// This resolv.conf file is shared by all containers of the same pod, and needs to be modified only once per pod.
	// we modify it when the pause container is created since it is the first container created in the pod since it holds
	// the networking namespace.
	if container.Name == PodInfraContainerName && utsMode != namespaceModeHost {
		err = addNDotsOption(containerInfo.ResolvConfPath)
		if err != nil {
			return kubecontainer.ContainerID{}, fmt.Errorf("addNDotsOption: %v", err)
		}
	}

	return id, err
}

func (dm *DockerManager) applyOOMScoreAdjIfNeeded(container *api.Container, containerInfo *dockertypes.ContainerJSON) error {
	// Compare current API version with expected api version.
	result, err := dm.checkDockerAPIVersion(dockerv110APIVersion)
	if err != nil {
		return fmt.Errorf("Failed to check docker api version: %v", err)
	}
	// If current api version is older than OOMScoreAdj requested, use the old way.
	if result < 0 {
		if err := dm.applyOOMScoreAdj(container, containerInfo); err != nil {
			return fmt.Errorf("Failed to apply oom-score-adj to container %q- %v", err, containerInfo.Name)
		}
	}

	return nil
}

func (dm *DockerManager) calculateOomScoreAdj(container *api.Container) int {
	// Set OOM score of the container based on the priority of the container.
	// Processes in lower-priority pods should be killed first if the system runs out of memory.
	// The main pod infrastructure container is considered high priority, since if it is killed the
	// whole pod will die.
	var oomScoreAdj int
	if container.Name == PodInfraContainerName {
		oomScoreAdj = qos.PodInfraOOMAdj
	} else {
		oomScoreAdj = qos.GetContainerOOMScoreAdjust(container, int64(dm.machineInfo.MemoryCapacity))

	}

	return oomScoreAdj
}

// versionInfo wraps api version and daemon version.
type versionInfo struct {
	apiVersion    kubecontainer.Version
	daemonVersion kubecontainer.Version
}

// checkDockerAPIVersion checks current docker API version against expected version.
// Return:
// 1 : newer than expected version
// -1: older than expected version
// 0 : same version
func (dm *DockerManager) checkDockerAPIVersion(expectedVersion string) (int, error) {

	value, err := dm.versionCache.Get(dm.machineInfo.MachineID)
	if err != nil {
		return 0, err
	}
	apiVersion := value.(versionInfo).apiVersion
	result, err := apiVersion.Compare(expectedVersion)
	if err != nil {
		return 0, fmt.Errorf("Failed to compare current docker api version %v with OOMScoreAdj supported Docker version %q - %v",
			apiVersion, expectedVersion, err)
	}
	return result, nil
}

func addNDotsOption(resolvFilePath string) error {
	if len(resolvFilePath) == 0 {
		glog.Errorf("ResolvConfPath is empty.")
		return nil
	}

	if _, err := os.Stat(resolvFilePath); os.IsNotExist(err) {
		return fmt.Errorf("ResolvConfPath %q does not exist", resolvFilePath)
	}

	glog.V(4).Infof("DNS ResolvConfPath exists: %s. Will attempt to add ndots option: %s", resolvFilePath, ndotsDNSOption)

	if err := appendToFile(resolvFilePath, ndotsDNSOption); err != nil {
		glog.Errorf("resolv.conf could not be updated: %v", err)
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
// If any error occurs in this function, it will return a brief error and a detailed error message.
func (dm *DockerManager) createPodInfraContainer(pod *api.Pod) (kubecontainer.DockerID, error, string) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("createPodInfraContainer").Observe(metrics.SinceInMicroseconds(start))
	}()
	// Use host networking if specified.
	netNamespace := ""
	var ports []api.ContainerPort

	if kubecontainer.IsHostNetworkPod(pod) {
		netNamespace = namespaceModeHost
	} else if dm.networkPlugin.Name() == "cni" || dm.networkPlugin.Name() == "kubenet" {
		netNamespace = "none"
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
		Env:             dm.podInfraContainerEnv,
	}

	// No pod secrets for the infra container.
	// The message isn't needed for the Infra container
	if err, msg := dm.imagePuller.PullImage(pod, container, nil); err != nil {
		return "", err, msg
	}

	// Currently we don't care about restart count of infra container, just set it to 0.
	id, err := dm.runContainerInPod(pod, container, netNamespace, getIPCMode(pod), getPidMode(pod), "", 0)
	if err != nil {
		return "", kubecontainer.ErrRunContainer, err.Error()
	}

	return kubecontainer.DockerID(id.ID), nil, ""
}

// Structure keeping information on changes that need to happen for a pod. The semantics is as follows:
// - startInfraContainer is true if new Infra Containers have to be started and old one (if running) killed.
//   Additionally if it is true then containersToKeep have to be empty
// - infraContainerId have to be set if and only if startInfraContainer is false. It stores dockerID of running Infra Container
// - containersToStart keeps indices of Specs of containers that have to be started and reasons why containers will be started.
// - containersToKeep stores mapping from dockerIDs of running containers to indices of their Specs for containers that
//   should be kept running. If startInfraContainer is false then it contains an entry for infraContainerId (mapped to -1).
//   It shouldn't be the case where containersToStart is empty and containersToKeep contains only infraContainerId. In such case
//   Infra Container should be killed, hence it's removed from this map.
// - all running containers which are NOT contained in containersToKeep should be killed.
type podContainerChangesSpec struct {
	StartInfraContainer bool
	InfraChanged        bool
	InfraContainerId    kubecontainer.DockerID
	ContainersToStart   map[int]string
	ContainersToKeep    map[kubecontainer.DockerID]int
}

func (dm *DockerManager) computePodContainerChanges(pod *api.Pod, podStatus *kubecontainer.PodStatus) (podContainerChangesSpec, error) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("computePodContainerChanges").Observe(metrics.SinceInMicroseconds(start))
	}()
	glog.V(5).Infof("Syncing Pod %q: %+v", format.Pod(pod), pod)

	containersToStart := make(map[int]string)
	containersToKeep := make(map[kubecontainer.DockerID]int)

	var err error
	var podInfraContainerID kubecontainer.DockerID
	var changed bool
	podInfraContainerStatus := podStatus.FindContainerStatusByName(PodInfraContainerName)
	if podInfraContainerStatus != nil && podInfraContainerStatus.State == kubecontainer.ContainerStateRunning {
		glog.V(4).Infof("Found pod infra container for %q", format.Pod(pod))
		changed, err = dm.podInfraContainerChanged(pod, podInfraContainerStatus)
		if err != nil {
			return podContainerChangesSpec{}, err
		}
	}

	createPodInfraContainer := true
	if podInfraContainerStatus == nil || podInfraContainerStatus.State != kubecontainer.ContainerStateRunning {
		glog.V(2).Infof("Need to restart pod infra container for %q because it is not found", format.Pod(pod))
	} else if changed {
		glog.V(2).Infof("Need to restart pod infra container for %q because it is changed", format.Pod(pod))
	} else {
		glog.V(4).Infof("Pod infra container looks good, keep it %q", format.Pod(pod))
		createPodInfraContainer = false
		podInfraContainerID = kubecontainer.DockerID(podInfraContainerStatus.ID.ID)
		containersToKeep[podInfraContainerID] = -1
	}

	for index, container := range pod.Spec.Containers {
		expectedHash := kubecontainer.HashContainer(&container)

		containerStatus := podStatus.FindContainerStatusByName(container.Name)
		if containerStatus == nil || containerStatus.State != kubecontainer.ContainerStateRunning {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, podStatus) {
				// If we are here it means that the container is dead and should be restarted, or never existed and should
				// be created. We may be inserting this ID again if the container has changed and it has
				// RestartPolicy::Always, but it's not a big deal.
				message := fmt.Sprintf("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				glog.V(3).Info(message)
				containersToStart[index] = message
			}
			continue
		}

		containerID := kubecontainer.DockerID(containerStatus.ID.ID)
		hash := containerStatus.Hash
		glog.V(3).Infof("pod %q container %q exists as %v", format.Pod(pod), container.Name, containerID)

		if createPodInfraContainer {
			// createPodInfraContainer == true and Container exists
			// If we're creating infra container everything will be killed anyway
			// If RestartPolicy is Always or OnFailure we restart containers that were running before we
			// killed them when restarting Infra Container.
			if pod.Spec.RestartPolicy != api.RestartPolicyNever {
				message := fmt.Sprintf("Infra Container is being recreated. %q will be restarted.", container.Name)
				glog.V(1).Info(message)
				containersToStart[index] = message
			}
			continue
		}

		// At this point, the container is running and pod infra container is good.
		// We will look for changes and check healthiness for the container.
		containerChanged := hash != 0 && hash != expectedHash
		if containerChanged {
			message := fmt.Sprintf("pod %q container %q hash changed (%d vs %d), it will be killed and re-created.", format.Pod(pod), container.Name, hash, expectedHash)
			glog.Info(message)
			containersToStart[index] = message
			continue
		}

		liveness, found := dm.livenessManager.Get(containerStatus.ID)
		if !found || liveness == proberesults.Success {
			containersToKeep[containerID] = index
			continue
		}
		if pod.Spec.RestartPolicy != api.RestartPolicyNever {
			message := fmt.Sprintf("pod %q container %q is unhealthy, it will be killed and re-created.", format.Pod(pod), container.Name)
			glog.Info(message)
			containersToStart[index] = message
		}
	}

	// After the loop one of the following should be true:
	// - createPodInfraContainer is true and containersToKeep is empty.
	// (In fact, when createPodInfraContainer is false, containersToKeep will not be touched).
	// - createPodInfraContainer is false and containersToKeep contains at least ID of Infra Container

	// If Infra container is the last running one, we don't want to keep it.
	if !createPodInfraContainer && len(containersToStart) == 0 && len(containersToKeep) == 1 {
		containersToKeep = make(map[kubecontainer.DockerID]int)
	}

	return podContainerChangesSpec{
		StartInfraContainer: createPodInfraContainer,
		InfraChanged:        changed,
		InfraContainerId:    podInfraContainerID,
		ContainersToStart:   containersToStart,
		ContainersToKeep:    containersToKeep,
	}, nil
}

// Sync the running pod to match the specified desired pod.
func (dm *DockerManager) SyncPod(pod *api.Pod, _ api.PodStatus, podStatus *kubecontainer.PodStatus, pullSecrets []api.Secret, backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
	start := time.Now()
	defer func() {
		metrics.ContainerManagerLatency.WithLabelValues("SyncPod").Observe(metrics.SinceInMicroseconds(start))
	}()

	containerChanges, err := dm.computePodContainerChanges(pod, podStatus)
	if err != nil {
		result.Fail(err)
		return
	}
	glog.V(3).Infof("Got container changes for pod %q: %+v", format.Pod(pod), containerChanges)

	if containerChanges.InfraChanged {
		ref, err := api.GetReference(pod)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %q: '%v'", format.Pod(pod), err)
		}
		dm.recorder.Eventf(ref, api.EventTypeNormal, "InfraChanged", "Pod infrastructure changed, it will be killed and re-created.")
	}
	if containerChanges.StartInfraContainer || (len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0) {
		if len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0 {
			glog.V(4).Infof("Killing Infra Container for %q because all other containers are dead.", format.Pod(pod))
		} else {
			glog.V(4).Infof("Killing Infra Container for %q, will start new one", format.Pod(pod))
		}

		// Killing phase: if we want to start new infra container, or nothing is running kill everything (including infra container)
		// TODO(random-liu): We'll use pod status directly in the future
		killResult := dm.killPodWithSyncResult(pod, kubecontainer.ConvertPodStatusToRunningPod(podStatus))
		result.AddPodSyncResult(killResult)
		if killResult.Error() != nil {
			return
		}
	} else {
		// Otherwise kill any running containers in this pod which are not specified as ones to keep.
		runningContainerStatues := podStatus.GetRunningContainerStatuses()
		for _, containerStatus := range runningContainerStatues {
			_, keep := containerChanges.ContainersToKeep[kubecontainer.DockerID(containerStatus.ID.ID)]
			if !keep {
				glog.V(3).Infof("Killing unwanted container %q(id=%q) for pod %q", containerStatus.Name, containerStatus.ID, format.Pod(pod))
				// attempt to find the appropriate container policy
				var podContainer *api.Container
				var killMessage string
				for i, c := range pod.Spec.Containers {
					if c.Name == containerStatus.Name {
						podContainer = &pod.Spec.Containers[i]
						killMessage = containerChanges.ContainersToStart[i]
						break
					}
				}
				killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, containerStatus.Name)
				result.AddSyncResult(killContainerResult)
				if err := dm.KillContainerInPod(containerStatus.ID, podContainer, pod, killMessage); err != nil {
					killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
					glog.Errorf("Error killing container %q(id=%q) for pod %q: %v", containerStatus.Name, containerStatus.ID, format.Pod(pod), err)
					return
				}
			}
		}
	}

	// We pass the value of the podIP down to runContainerInPod, which in turn
	// passes it to various other functions, in order to facilitate
	// functionality that requires this value (hosts file and downward API)
	// and avoid races determining the pod IP in cases where a container
	// requires restart but the podIP isn't in the status manager yet.
	//
	// We default to the IP in the passed-in pod status, and overwrite it if the
	// infra container needs to be (re)started.
	podIP := ""
	if podStatus != nil {
		podIP = podStatus.IP
	}

	// If we should create infra container then we do it first.
	podInfraContainerID := containerChanges.InfraContainerId
	if containerChanges.StartInfraContainer && (len(containerChanges.ContainersToStart) > 0) {
		glog.V(4).Infof("Creating pod infra container for %q", format.Pod(pod))
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, PodInfraContainerName)
		result.AddSyncResult(startContainerResult)
		var msg string
		podInfraContainerID, err, msg = dm.createPodInfraContainer(pod)
		if err != nil {
			startContainerResult.Fail(err, msg)
			glog.Errorf("Failed to create pod infra container: %v; Skipping pod %q", err, format.Pod(pod))
			return
		}

		setupNetworkResult := kubecontainer.NewSyncResult(kubecontainer.SetupNetwork, kubecontainer.GetPodFullName(pod))
		result.AddSyncResult(setupNetworkResult)
		if !kubecontainer.IsHostNetworkPod(pod) {
			glog.V(3).Infof("Calling network plugin %s to setup pod for %s", dm.networkPlugin.Name(), format.Pod(pod))
			err = dm.networkPlugin.SetUpPod(pod.Namespace, pod.Name, podInfraContainerID.ContainerID())
			if err != nil {
				// TODO: (random-liu) There shouldn't be "Skipping pod" in sync result message
				message := fmt.Sprintf("Failed to setup network for pod %q using network plugins %q: %v; Skipping pod", format.Pod(pod), dm.networkPlugin.Name(), err)
				setupNetworkResult.Fail(kubecontainer.ErrSetupNetwork, message)
				glog.Error(message)

				// Delete infra container
				killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, PodInfraContainerName)
				result.AddSyncResult(killContainerResult)
				if delErr := dm.KillContainerInPod(kubecontainer.ContainerID{
					ID:   string(podInfraContainerID),
					Type: "docker"}, nil, pod, message); delErr != nil {
					killContainerResult.Fail(kubecontainer.ErrKillContainer, delErr.Error())
					glog.Warningf("Clear infra container failed for pod %q: %v", format.Pod(pod), delErr)
				}
				return
			}

			// Setup the host interface unless the pod is on the host's network (FIXME: move to networkPlugin when ready)
			podInfraContainer, err := dm.client.InspectContainer(string(podInfraContainerID))
			if err != nil {
				glog.Errorf("Failed to inspect pod infra container: %v; Skipping pod %q", err, format.Pod(pod))
				result.Fail(err)
				return
			}

			if dm.configureHairpinMode {
				if err = hairpin.SetUpContainer(podInfraContainer.State.Pid, network.DefaultInterfaceName); err != nil {
					glog.Warningf("Hairpin setup failed for pod %q: %v", format.Pod(pod), err)
				}
			}

			// Overwrite the podIP passed in the pod status, since we just started the infra container.
			podIP = dm.determineContainerIP(pod.Name, pod.Namespace, podInfraContainer)
		}
	}

	// Start everything
	for idx := range containerChanges.ContainersToStart {
		container := &pod.Spec.Containers[idx]
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, container.Name)
		result.AddSyncResult(startContainerResult)

		// containerChanges.StartInfraContainer causes the containers to be restarted for config reasons
		// ignore backoff
		if !containerChanges.StartInfraContainer {
			isInBackOff, err, msg := dm.doBackOff(pod, container, podStatus, backOff)
			if isInBackOff {
				startContainerResult.Fail(err, msg)
				glog.V(4).Infof("Backing Off restarting container %+v in pod %v", container, format.Pod(pod))
				continue
			}
		}
		glog.V(4).Infof("Creating container %+v in pod %v", container, format.Pod(pod))
		err, msg := dm.imagePuller.PullImage(pod, container, pullSecrets)
		if err != nil {
			startContainerResult.Fail(err, msg)
			continue
		}

		if container.SecurityContext != nil && container.SecurityContext.RunAsNonRoot != nil && *container.SecurityContext.RunAsNonRoot {
			err := dm.verifyNonRoot(container)
			if err != nil {
				startContainerResult.Fail(kubecontainer.ErrVerifyNonRoot, err.Error())
				glog.Errorf("Error running pod %q container %q: %v", format.Pod(pod), container.Name, err)
				continue
			}
		}
		// For a new container, the RestartCount should be 0
		restartCount := 0
		containerStatus := podStatus.FindContainerStatusByName(container.Name)
		if containerStatus != nil {
			restartCount = containerStatus.RestartCount + 1
		}

		// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
		// Note: when configuring the pod's containers anything that can be configured by pointing
		// to the namespace of the infra container should use namespaceMode.  This includes things like the net namespace
		// and IPC namespace.  PID mode cannot point to another container right now.
		// See createPodInfraContainer for infra container setup.
		namespaceMode := fmt.Sprintf("container:%v", podInfraContainerID)
		_, err = dm.runContainerInPod(pod, container, namespaceMode, namespaceMode, getPidMode(pod), podIP, restartCount)
		if err != nil {
			startContainerResult.Fail(kubecontainer.ErrRunContainer, err.Error())
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running pod %q container %q: %v", format.Pod(pod), container.Name, err)
			continue
		}
		// Successfully started the container; clear the entry in the failure
	}
	return
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
		return fmt.Errorf("can't tell if image runs as root: %v", err)
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
		return false, fmt.Errorf("non-numeric user (%s) is not allowed", user)
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

// If all instances of a container are garbage collected, doBackOff will also return false, which means the container may be restarted before the
// backoff deadline. However, because that won't cause error and the chance is really slim, we can just ignore it for now.
// If a container is still in backoff, the function will return a brief backoff error and a detailed error message.
func (dm *DockerManager) doBackOff(pod *api.Pod, container *api.Container, podStatus *kubecontainer.PodStatus, backOff *flowcontrol.Backoff) (bool, error, string) {
	var cStatus *kubecontainer.ContainerStatus
	// Use the finished time of the latest exited container as the start point to calculate whether to do back-off.
	// TODO(random-liu): Better define backoff start point; add unit and e2e test after we finalize this. (See github issue #22240)
	for _, c := range podStatus.ContainerStatuses {
		if c.Name == container.Name && c.State == kubecontainer.ContainerStateExited {
			cStatus = c
			break
		}
	}
	if cStatus != nil {
		ts := cStatus.FinishedAt
		// found a container that requires backoff
		dockerName := KubeletContainerName{
			PodFullName:   kubecontainer.GetPodFullName(pod),
			PodUID:        pod.UID,
			ContainerName: container.Name,
		}
		stableName, _, _ := BuildDockerName(dockerName, container)
		if backOff.IsInBackOffSince(stableName, ts) {
			if ref, err := kubecontainer.GenerateContainerRef(pod, container); err == nil {
				dm.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.BackOffStartContainer, "Back-off restarting failed docker container")
			}
			err := fmt.Errorf("Back-off %s restarting failed container=%s pod=%s", backOff.Get(stableName), container.Name, format.Pod(pod))
			glog.Infof("%s", err.Error())
			return true, kubecontainer.ErrCrashLoopBackOff, err.Error()
		}
		backOff.Next(stableName, ts)
	}
	return false, nil, ""
}

// getPidMode returns the pid mode to use on the docker container based on pod.Spec.HostPID.
func getPidMode(pod *api.Pod) string {
	pidMode := ""
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostPID {
		pidMode = namespaceModeHost
	}
	return pidMode
}

// getIPCMode returns the ipc mode to use on the docker container based on pod.Spec.HostIPC.
func getIPCMode(pod *api.Pod) string {
	ipcMode := ""
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostIPC {
		ipcMode = namespaceModeHost
	}
	return ipcMode
}

// GetNetNS returns the network namespace path for the given container
func (dm *DockerManager) GetNetNS(containerID kubecontainer.ContainerID) (string, error) {
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

func (dm *DockerManager) GetPodStatus(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	podStatus := &kubecontainer.PodStatus{ID: uid, Name: name, Namespace: namespace}
	// Now we retain restart count of container as a docker label. Each time a container
	// restarts, pod will read the restart count from the registered dead container, increment
	// it to get the new restart count, and then add a label with the new restart count on
	// the newly started container.
	// However, there are some limitations of this method:
	//	1. When all dead containers were garbage collected, the container status could
	//	not get the historical value and would be *inaccurate*. Fortunately, the chance
	//	is really slim.
	//	2. When working with old version containers which have no restart count label,
	//	we can only assume their restart count is 0.
	// Anyhow, we only promised "best-effort" restart count reporting, we can just ignore
	// these limitations now.
	var containerStatuses []*kubecontainer.ContainerStatus
	// We have added labels like pod name and pod namespace, it seems that we can do filtered list here.
	// However, there may be some old containers without these labels, so at least now we can't do that.
	// TODO(random-liu): Do only one list and pass in the list result in the future
	// TODO(random-liu): Add filter when we are sure that all the containers have the labels
	containers, err := dm.client.ListContainers(dockertypes.ContainerListOptions{All: true})
	if err != nil {
		return podStatus, err
	}
	// Loop through list of running and exited docker containers to construct
	// the statuses. We assume docker returns a list of containers sorted in
	// reverse by time.
	// TODO: optimization: set maximum number of containers per container name to examine.
	for _, c := range containers {
		if len(c.Names) == 0 {
			continue
		}
		dockerName, _, err := ParseDockerName(c.Names[0])
		if err != nil {
			continue
		}
		if dockerName.PodUID != uid {
			continue
		}
		result, ip, err := dm.inspectContainer(c.ID, name, namespace)
		if err != nil {
			if _, ok := err.(containerNotFoundError); ok {
				// https://github.com/kubernetes/kubernetes/issues/22541
				// Sometimes when docker's state is corrupt, a container can be listed
				// but couldn't be inspected. We fake a status for this container so
				// that we can still return a status for the pod to sync.
				result = &kubecontainer.ContainerStatus{
					ID:    kubecontainer.DockerID(c.ID).ContainerID(),
					Name:  dockerName.ContainerName,
					State: kubecontainer.ContainerStateUnknown,
				}
				glog.Errorf("Unable to inspect container %q: %v", c.ID, err)
			} else {
				return podStatus, err
			}
		}
		containerStatuses = append(containerStatuses, result)
		if ip != "" {
			podStatus.IP = ip
		}
	}

	podStatus.ContainerStatuses = containerStatuses
	return podStatus, nil
}

// getVersionInfo returns apiVersion & daemonVersion of docker runtime
func (dm *DockerManager) getVersionInfo() (versionInfo, error) {
	apiVersion, err := dm.APIVersion()
	if err != nil {
		return versionInfo{}, err
	}
	daemonVersion, err := dm.Version()
	if err != nil {
		return versionInfo{}, err
	}
	return versionInfo{
		apiVersion:    apiVersion,
		daemonVersion: daemonVersion,
	}, nil
}
