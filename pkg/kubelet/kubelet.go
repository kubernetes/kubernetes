/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"
	"io"
	"net/http"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const defaultChanSize = 1024

// taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
const minShares = 2
const sharesPerCPU = 1024
const milliCPUToCPU = 1000

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	SyncPods([]api.BoundPod) error
}

type volumeMap map[string]volume.Interface

// New creates a new Kubelet for use in main
func NewMainKubelet(
	hn string,
	dc dockertools.DockerInterface,
	ec tools.EtcdClient,
	rd string,
	ni string,
	ri time.Duration,
	pullQPS float32,
	pullBurst int) *Kubelet {
	return &Kubelet{
		hostname:              hn,
		dockerClient:          dc,
		etcdClient:            ec,
		rootDirectory:         rd,
		resyncInterval:        ri,
		networkContainerImage: ni,
		podWorkers:            newPodWorkers(),
		runner:                dockertools.NewDockerContainerCommandRunner(),
		httpClient:            &http.Client{},
		pullQPS:               pullQPS,
		pullBurst:             pullBurst,
	}
}

// NewIntegrationTestKubelet creates a new Kubelet for use in integration tests.
// TODO: add more integration tests, and expand parameter list as needed.
func NewIntegrationTestKubelet(hn string, rd string, dc dockertools.DockerInterface) *Kubelet {
	return &Kubelet{
		hostname:              hn,
		dockerClient:          dc,
		rootDirectory:         rd,
		dockerPuller:          &dockertools.FakeDockerPuller{},
		networkContainerImage: NetworkContainerImage,
		resyncInterval:        3 * time.Second,
		podWorkers:            newPodWorkers(),
	}
}

type httpGetter interface {
	Get(url string) (*http.Response, error)
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	hostname              string
	dockerClient          dockertools.DockerInterface
	rootDirectory         string
	networkContainerImage string
	podWorkers            *podWorkers
	resyncInterval        time.Duration
	pods                  []api.BoundPod

	// Optional, no events will be sent without it
	etcdClient tools.EtcdClient
	// Optional, defaults to simple implementaiton
	healthChecker health.HealthChecker
	// Optional, defaults to simple Docker implementation
	dockerPuller dockertools.DockerPuller
	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner dockertools.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient httpGetter
	// Optional, maximum pull QPS from the docker registry, 0.0 means unlimited.
	pullQPS float32
	// Optional, maximum burst QPS from the docker registry, must be positive if QPS is > 0.0
	pullBurst int

	// Optional, no statistics will be available if omitted
	cadvisorClient cadvisorInterface
	cadvisorLock   sync.RWMutex
}

// SetCadvisorClient sets the cadvisor client in a thread-safe way.
func (kl *Kubelet) SetCadvisorClient(c cadvisorInterface) {
	kl.cadvisorLock.Lock()
	defer kl.cadvisorLock.Unlock()
	kl.cadvisorClient = c
}

// GetCadvisorClient gets the cadvisor client.
func (kl *Kubelet) GetCadvisorClient() cadvisorInterface {
	kl.cadvisorLock.RLock()
	defer kl.cadvisorLock.RUnlock()
	return kl.cadvisorClient
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.dockerPuller == nil {
		kl.dockerPuller = dockertools.NewDockerPuller(kl.dockerClient, kl.pullQPS, kl.pullBurst)
	}
	if kl.healthChecker == nil {
		kl.healthChecker = health.NewHealthChecker()
	}
	kl.syncLoop(updates, kl)
}

// Per-pod workers.
type podWorkers struct {
	lock sync.Mutex

	// Set of pods with existing workers.
	workers util.StringSet
}

func newPodWorkers() *podWorkers {
	return &podWorkers{
		workers: util.NewStringSet(),
	}
}

// Runs a worker for "podFullName" asynchronously with the specified "action".
// If the worker for the "podFullName" is already running, functions as a no-op.
func (self *podWorkers) Run(podFullName string, action func()) {
	self.lock.Lock()
	defer self.lock.Unlock()

	// This worker is already running, let it finish.
	if self.workers.Has(podFullName) {
		return
	}
	self.workers.Insert(podFullName)

	// Run worker async.
	go func() {
		defer util.HandleCrash()
		action()

		self.lock.Lock()
		defer self.lock.Unlock()
		self.workers.Delete(podFullName)
	}()
}

// LogEvent reports an event.
func (kl *Kubelet) LogEvent(event *api.Event) error {
	return nil
}

func makeEnvironmentVariables(container *api.Container) []string {
	var result []string
	for _, value := range container.Env {
		result = append(result, fmt.Sprintf("%s=%s", value.Name, value.Value))
	}
	return result
}

func makeBinds(pod *api.BoundPod, container *api.Container, podVolumes volumeMap) []string {
	binds := []string{}
	for _, mount := range container.VolumeMounts {
		vol, ok := podVolumes[mount.Name]
		if !ok {
			continue
		}
		b := fmt.Sprintf("%s:%s", vol.GetPath(), mount.MountPath)
		if mount.ReadOnly {
			b += ":ro"
		}
		binds = append(binds, b)
	}
	return binds
}

func makePortsAndBindings(container *api.Container) (map[docker.Port]struct{}, map[docker.Port][]docker.PortBinding) {
	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range container.Ports {
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
			glog.Warningf("Unknown protocol '%s': defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			{
				HostPort: strconv.Itoa(exteriorPort),
				HostIp:   port.HostIP,
			},
		}
	}
	return exposedPorts, portBindings
}

func milliCPUToShares(milliCPU int) int {
	if milliCPU == 0 {
		// zero milliCPU means unset. Use kernel default.
		return 0
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

func (kl *Kubelet) mountExternalVolumes(pod *api.BoundPod) (volumeMap, error) {
	podVolumes := make(volumeMap)
	for _, vol := range pod.Spec.Volumes {
		extVolume, err := volume.CreateVolumeBuilder(&vol, pod.Name, kl.rootDirectory)
		if err != nil {
			return nil, err
		}
		// TODO(jonesdl) When the default volume behavior is no longer supported, this case
		// should never occur and an error should be thrown instead.
		if extVolume == nil {
			continue
		}
		podVolumes[vol.Name] = extVolume
		err = extVolume.SetUp()
		if err != nil {
			return nil, err
		}
	}
	return podVolumes, nil
}

// A basic interface that knows how to execute handlers
type actionHandler interface {
	Run(podFullName, uuid string, container *api.Container, handler *api.Handler) error
}

func (kl *Kubelet) newActionHandler(handler *api.Handler) actionHandler {
	switch {
	case handler.Exec != nil:
		return &execActionHandler{kubelet: kl}
	case handler.HTTPGet != nil:
		return &httpActionHandler{client: kl.httpClient, kubelet: kl}
	default:
		glog.Errorf("Invalid handler: %v", handler)
		return nil
	}
}

func (kl *Kubelet) runHandler(podFullName, uuid string, container *api.Container, handler *api.Handler) error {
	actionHandler := kl.newActionHandler(handler)
	if actionHandler == nil {
		return fmt.Errorf("invalid handler")
	}
	return actionHandler.Run(podFullName, uuid, container, handler)
}

// Run a single container from a pod. Returns the docker container ID
func (kl *Kubelet) runContainer(pod *api.BoundPod, container *api.Container, podVolumes volumeMap, netMode string) (id dockertools.DockerID, err error) {
	envVariables := makeEnvironmentVariables(container)
	binds := makeBinds(pod, container, podVolumes)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: dockertools.BuildDockerName(pod.UID, GetPodFullName(pod), container),
		Config: &docker.Config{
			Cmd:          container.Command,
			Env:          envVariables,
			ExposedPorts: exposedPorts,
			Hostname:     pod.Name,
			Image:        container.Image,
			Memory:       int64(container.Memory),
			CpuShares:    int64(milliCPUToShares(container.CPU)),
			WorkingDir:   container.WorkingDir,
		},
	}
	dockerContainer, err := kl.dockerClient.CreateContainer(opts)
	if err != nil {
		return "", err
	}
	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = container.Privileged
	} else if container.Privileged {
		return "", fmt.Errorf("Container requested privileged mode, but it is disallowed globally.")
	}
	err = kl.dockerClient.StartContainer(dockerContainer.ID, &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
		Privileged:   privileged,
	})
	if err == nil && container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := kl.runHandler(GetPodFullName(pod), pod.UID, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			kl.killContainerByID(dockerContainer.ID, "")
			return dockertools.DockerID(""), fmt.Errorf("failed to call event handler: %v", handlerErr)
		}
	}
	return dockertools.DockerID(dockerContainer.ID), err
}

// Kill a docker container
func (kl *Kubelet) killContainer(dockerContainer *docker.APIContainers) error {
	return kl.killContainerByID(dockerContainer.ID, dockerContainer.Names[0])
}

func (kl *Kubelet) killContainerByID(ID, name string) error {
	glog.V(2).Infof("Killing: %s", ID)
	err := kl.dockerClient.StopContainer(ID, 10)
	if len(name) == 0 {
		return err
	}

	// TODO(lavalamp): restore event logging:
	// podFullName, uuid, containerName, _ := dockertools.ParseDockerName(name)
	// kl.LogEvent(&api.Event{})

	return err
}

const (
	networkContainerName  = "net"
	NetworkContainerImage = "kubernetes/pause:latest"
)

// createNetworkContainer starts the network container for a pod. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createNetworkContainer(pod *api.BoundPod) (dockertools.DockerID, error) {
	var ports []api.Port
	// Docker only exports ports from the network container.  Let's
	// collect all of the relevant ports and export them.
	for _, container := range pod.Spec.Containers {
		ports = append(ports, container.Ports...)
	}
	container := &api.Container{
		Name:  networkContainerName,
		Image: kl.networkContainerImage,
		Ports: ports,
	}
	// TODO: make this a TTL based pull (if image older than X policy, pull)
	ok, err := kl.dockerPuller.IsImagePresent(container.Image)
	if err != nil {
		return "", err
	}
	if !ok {
		if err := kl.dockerPuller.Pull(container.Image); err != nil {
			return "", err
		}
	}
	return kl.runContainer(pod, container, nil, "")
}

// Kill all containers in a pod.  Returns the number of containers deleted and an error if one occurs.
func (kl *Kubelet) killContainersInPod(pod *api.BoundPod, dockerContainers dockertools.DockerContainers) (int, error) {
	podFullName := GetPodFullName(pod)

	count := 0
	errs := make(chan error, len(pod.Spec.Containers))
	wg := sync.WaitGroup{}
	for _, container := range pod.Spec.Containers {
		// TODO: Consider being more aggressive: kill all containers with this pod UID, period.
		if dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, pod.UID, container.Name); found {
			count++
			wg.Add(1)
			go func() {
				err := kl.killContainer(dockerContainer)
				if err != nil {
					glog.Errorf("Failed to delete container. (%v)  Skipping pod %s", err, podFullName)
					errs <- err
				}
				wg.Done()
			}()
		}
	}
	wg.Wait()
	close(errs)
	if len(errs) > 0 {
		errList := []error{}
		for err := range errs {
			errList = append(errList, err)
		}
		return -1, fmt.Errorf("failed to delete containers (%v)", errList)
	}
	return count, nil
}

type empty struct{}

func (kl *Kubelet) syncPod(pod *api.BoundPod, dockerContainers dockertools.DockerContainers) error {
	podFullName := GetPodFullName(pod)
	uuid := pod.UID
	containersToKeep := make(map[dockertools.DockerID]empty)
	killedContainers := make(map[dockertools.DockerID]empty)

	// Make sure we have a network container
	var netID dockertools.DockerID
	if netDockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, networkContainerName); found {
		netID = dockertools.DockerID(netDockerContainer.ID)
	} else {
		glog.V(3).Infof("Network container doesn't exist for pod %q, re-creating the pod", podFullName)
		count, err := kl.killContainersInPod(pod, dockerContainers)
		if err != nil {
			return err
		}
		netID, err = kl.createNetworkContainer(pod)
		if err != nil {
			glog.Errorf("Failed to introspect network container. (%v)  Skipping pod %s", err, podFullName)
			return err
		}
		if count > 0 {
			// Re-list everything, otherwise we'll think we're ok.
			dockerContainers, err = dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
			if err != nil {
				glog.Errorf("Error listing containers %#v", dockerContainers)
				return err
			}
		}
	}
	containersToKeep[netID] = empty{}

	podVolumes, err := kl.mountExternalVolumes(pod)
	if err != nil {
		glog.Errorf("Unable to mount volumes for pod %s: (%v), skipping pod", podFullName, err)
		return err
	}

	podState := api.PodState{}
	info, err := kl.GetPodInfo(podFullName, uuid)
	if err != nil {
		glog.Errorf("Unable to get pod with name %s and uuid %s info, health checks may be invalid", podFullName, uuid)
	}
	netInfo, found := info[networkContainerName]
	if found {
		podState.PodIP = netInfo.PodIP
	}

	for _, container := range pod.Spec.Containers {
		expectedHash := dockertools.HashContainer(&container)
		if dockerContainer, found, hash := dockerContainers.FindPodContainer(podFullName, uuid, container.Name); found {
			containerID := dockertools.DockerID(dockerContainer.ID)
			glog.V(3).Infof("pod %s container %s exists as %v", podFullName, container.Name, containerID)

			// look for changes in the container.
			if hash == 0 || hash == expectedHash {
				// TODO: This should probably be separated out into a separate goroutine.
				healthy, err := kl.healthy(podFullName, uuid, podState, container, dockerContainer)
				if err != nil {
					glog.V(1).Infof("health check errored: %v", err)
					containersToKeep[containerID] = empty{}
					continue
				}
				if healthy == health.Healthy {
					containersToKeep[containerID] = empty{}
					continue
				}
				glog.V(1).Infof("pod %s container %s is unhealthy.", podFullName, container.Name, healthy)
			} else {
				glog.V(3).Infof("container hash changed %d vs %d.", hash, expectedHash)
			}
			if err := kl.killContainer(dockerContainer); err != nil {
				glog.V(1).Infof("Failed to kill container %s: %v", dockerContainer.ID, err)
				continue
			}
			killedContainers[containerID] = empty{}
		}

		// Check RestartPolicy for container
		recentContainers, err := dockertools.GetRecentDockerContainersWithNameAndUUID(kl.dockerClient, podFullName, uuid, container.Name)
		if err != nil {
			glog.Errorf("Error listing recent containers with name and uuid:%s--%s--%s", podFullName, uuid, container.Name)
			// TODO(dawnchen): error handling here?
		}

		if len(recentContainers) > 0 && pod.Spec.RestartPolicy.Always == nil {
			if pod.Spec.RestartPolicy.Never != nil {
				glog.V(3).Infof("Already ran container with name %s--%s--%s, do nothing",
					podFullName, uuid, container.Name)
				continue
			}
			if pod.Spec.RestartPolicy.OnFailure != nil {
				// Check the exit code of last run
				if recentContainers[0].State.ExitCode == 0 {
					glog.V(3).Infof("Already successfully ran container with name %s--%s--%s, do nothing",
						podFullName, uuid, container.Name)
					continue
				}
			}
		}

		glog.V(3).Infof("Container with name %s--%s--%s doesn't exist, creating %#v", podFullName, uuid, container.Name, container)
		if !api.IsPullNever(container.ImagePullPolicy) {
			present, err := kl.dockerPuller.IsImagePresent(container.Image)
			if err != nil {
				glog.Errorf("Failed to inspect image: %s: %#v skipping pod %s container %s", container.Image, err, podFullName, container.Name)
				continue
			}
			if api.IsPullAlways(container.ImagePullPolicy) || !present {
				if err := kl.dockerPuller.Pull(container.Image); err != nil {
					glog.Errorf("Failed to pull image %s: %v skipping pod %s container %s.", container.Image, err, podFullName, container.Name)
					continue
				}
			}
		}
		// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
		containerID, err := kl.runContainer(pod, &container, podVolumes, "container:"+string(netID))
		if err != nil {
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running pod %s container %s: %v", podFullName, container.Name, err)
			continue
		}
		containersToKeep[containerID] = empty{}
	}

	// Kill any containers in this pod which were not identified above (guards against duplicates).
	for id, container := range dockerContainers {
		curPodFullName, curUUID, _, _ := dockertools.ParseDockerName(container.Names[0])
		if curPodFullName == podFullName && curUUID == uuid {
			// Don't kill containers we want to keep or those we already killed.
			_, keep := containersToKeep[id]
			_, killed := killedContainers[id]
			if !keep && !killed {
				glog.V(1).Infof("Killing unwanted container in pod %q: %+v", curUUID, container)
				err = kl.killContainer(container)
				if err != nil {
					glog.Errorf("Error killing container: %v", err)
				}
			}
		}
	}

	return nil
}

type podContainer struct {
	podFullName   string
	uuid          string
	containerName string
}

// Stores all volumes defined by the set of pods into a map.
// Keys for each entry are in the format (POD_ID)/(VOLUME_NAME)
func getDesiredVolumes(pods []api.BoundPod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			identifier := path.Join(pod.Name, volume.Name)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
func (kl *Kubelet) reconcileVolumes(pods []api.BoundPod) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := volume.GetCurrentVolumes(kl.rootDirectory)
	for name, vol := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
			//TODO (jonesdl) We should somehow differentiate between volumes that are supposed
			//to be deleted and volumes that are leftover after a crash.
			glog.Warningf("Orphaned volume %s found, tearing down volume", name)
			//TODO (jonesdl) This should not block other kubelet synchronization procedures
			err := vol.TearDown()
			if err != nil {
				glog.Errorf("Could not tear down volume %s (%s)", name, err)
			}
		}
	}
	return nil
}

// SyncPods synchronizes the configured list of pods (desired state) with the host current state.
func (kl *Kubelet) SyncPods(pods []api.BoundPod) error {
	glog.V(4).Infof("Desired: %#v", pods)
	var err error
	desiredContainers := make(map[podContainer]empty)
	desiredPods := make(map[string]empty)

	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		glog.Errorf("Error listing containers: %#v", dockerContainers)
		return err
	}

	// Check for any containers that need starting
	for ix := range pods {
		pod := &pods[ix]
		podFullName := GetPodFullName(pod)
		uuid := pod.UID
		desiredPods[uuid] = empty{}

		// Add all containers (including net) to the map.
		desiredContainers[podContainer{podFullName, uuid, networkContainerName}] = empty{}
		for _, cont := range pod.Spec.Containers {
			desiredContainers[podContainer{podFullName, uuid, cont.Name}] = empty{}
		}

		// Run the sync in an async manifest worker.
		kl.podWorkers.Run(podFullName, func() {
			err := kl.syncPod(pod, dockerContainers)
			if err != nil {
				glog.Errorf("Error syncing pod, skipping: %s", err)
			}
		})
	}

	// Kill any containers we don't need.
	for _, container := range dockerContainers {
		// Don't kill containers that are in the desired pods.
		podFullName, uuid, containerName, _ := dockertools.ParseDockerName(container.Names[0])
		if _, found := desiredPods[uuid]; found {
			// syncPod() will handle this one.
			continue
		}
		pc := podContainer{podFullName, uuid, containerName}
		if _, ok := desiredContainers[pc]; !ok {
			glog.V(1).Infof("Killing unwanted container %+v", pc)
			err = kl.killContainer(container)
			if err != nil {
				glog.Errorf("Error killing container %+v: %s", pc, err)
			}
		}
	}

	// Remove any orphaned volumes.
	kl.reconcileVolumes(pods)

	return err
}

// filterHostPortConflicts removes pods that conflict on Port.HostPort values
func filterHostPortConflicts(pods []api.BoundPod) []api.BoundPod {
	filtered := []api.BoundPod{}
	ports := map[int]bool{}
	extract := func(p *api.Port) int { return p.HostPort }
	for i := range pods {
		pod := &pods[i]
		if errs := validation.AccumulateUniquePorts(pod.Spec.Containers, ports, extract); len(errs) != 0 {
			glog.Warningf("Pod %s: HostPort is already allocated, ignoring: %s", GetPodFullName(pod), errs)
			continue
		}
		filtered = append(filtered, *pod)
	}

	return filtered
}

// syncLoop is the main loop for processing changes. It watches for changes from
// four channels (file, etcd, server, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds. Never returns.
func (kl *Kubelet) syncLoop(updates <-chan PodUpdate, handler SyncHandler) {
	for {
		select {
		case u := <-updates:
			switch u.Op {
			case SET, UPDATE:
				glog.V(3).Infof("Containers changed")
				kl.pods = u.Pods
				kl.pods = filterHostPortConflicts(kl.pods)

			default:
				panic("syncLoop does not support incremental changes")
			}
		case <-time.After(kl.resyncInterval):
			glog.V(4).Infof("Periodic sync")
			if kl.pods == nil {
				continue
			}
		}

		err := handler.SyncPods(kl.pods)
		if err != nil {
			glog.Errorf("Couldn't sync containers: %s", err)
		}
	}
}

// GetKubeletContainerLogs returns logs from the container
// The second parameter of GetPodInfo and FindPodContainer methods represents pod UUID, which is allowed to be blank
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	_, err := kl.GetPodInfo(podFullName, "")
	if err == dockertools.ErrNoContainersInPod {
		return fmt.Errorf("Pod not found (%s)\n", podFullName)
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, true)
	if err != nil {
		return err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, "", containerName)
	if !found {
		return fmt.Errorf("Container not found (%s)\n", containerName)
	}
	return dockertools.GetKubeletDockerContainerLogs(kl.dockerClient, dockerContainer.ID, tail, follow, stdout, stderr)
}

// GetPodInfo returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodInfo(podFullName, uuid string) (api.PodInfo, error) {
	var manifest api.PodSpec
	for _, pod := range kl.pods {
		if GetPodFullName(&pod) == podFullName {
			manifest = pod.Spec
			break
		}
	}
	return dockertools.GetDockerPodInfo(kl.dockerClient, manifest, podFullName, uuid)
}

func (kl *Kubelet) healthy(podFullName, podUUID string, currentState api.PodState, container api.Container, dockerContainer *docker.APIContainers) (health.Status, error) {
	// Give the container 60 seconds to start up.
	if container.LivenessProbe == nil {
		return health.Healthy, nil
	}
	if time.Now().Unix()-dockerContainer.Created < container.LivenessProbe.InitialDelaySeconds {
		return health.Healthy, nil
	}
	if kl.healthChecker == nil {
		return health.Healthy, nil
	}
	return kl.healthChecker.HealthCheck(podFullName, podUUID, currentState, container)
}

// Returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}

// Run a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName, uuid, container string, cmd []string) ([]byte, error) {
	if kl.runner == nil {
		return nil, fmt.Errorf("no runner specified.")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return nil, err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, container)
	if !found {
		return nil, fmt.Errorf("container not found (%s)", container)
	}
	return kl.runner.RunInContainer(dockerContainer.ID, cmd)
}
