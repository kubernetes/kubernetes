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
	"encoding/json"
	"errors"
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
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/info"
)

const defaultChanSize = 1024

// taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
const minShares = 2
const sharesPerCPU = 1024
const milliCPUToCPU = 1000

// CadvisorInterface is an abstract interface for testability.  It abstracts the interface of "github.com/google/cadvisor/client".Client.
type CadvisorInterface interface {
	ContainerInfo(name string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	MachineInfo() (*info.MachineInfo, error)
}

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	SyncPods([]Pod) error
}

type volumeMap map[string]volume.Interface

// New creates a new Kubelet for use in main
func NewMainKubelet(
	hn string,
	dc dockertools.DockerInterface,
	cc CadvisorInterface,
	ec tools.EtcdClient,
	rd string,
	ri time.Duration) *Kubelet {
	return &Kubelet{
		hostname:       hn,
		dockerClient:   dc,
		cadvisorClient: cc,
		etcdClient:     ec,
		rootDirectory:  rd,
		resyncInterval: ri,
		podWorkers:     newPodWorkers(),
		runner:         dockertools.NewDockerContainerCommandRunner(),
		httpClient:     &http.Client{},
	}
}

// NewIntegrationTestKubelet creates a new Kubelet for use in integration tests.
// TODO: add more integration tests, and expand parameter list as needed.
func NewIntegrationTestKubelet(hn string, dc dockertools.DockerInterface) *Kubelet {
	return &Kubelet{
		hostname:       hn,
		dockerClient:   dc,
		dockerPuller:   &dockertools.FakeDockerPuller{},
		resyncInterval: 3 * time.Second,
		podWorkers:     newPodWorkers(),
	}
}

type httpGetInterface interface {
	Get(url string) (*http.Response, error)
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	hostname       string
	dockerClient   dockertools.DockerInterface
	rootDirectory  string
	podWorkers     podWorkers
	resyncInterval time.Duration

	// Optional, no events will be sent without it
	etcdClient tools.EtcdClient
	// Optional, no statistics will be available if omitted
	cadvisorClient CadvisorInterface
	// Optional, defaults to simple implementaiton
	healthChecker health.HealthChecker
	// Optional, defaults to simple Docker implementation
	dockerPuller dockertools.DockerPuller
	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner dockertools.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient httpGetInterface
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.dockerPuller == nil {
		kl.dockerPuller = dockertools.NewDockerPuller(kl.dockerClient)
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

func newPodWorkers() podWorkers {
	return podWorkers{
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

// LogEvent logs an event to the etcd backend.
func (kl *Kubelet) LogEvent(event *api.Event) error {
	if kl.etcdClient == nil {
		return fmt.Errorf("no etcd client connection")
	}
	event.Timestamp = time.Now().Unix()
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	var response *etcd.Response
	response, err = kl.etcdClient.AddChild(fmt.Sprintf("/events/%s", event.Container.Name), string(data), 60*60*48 /* 2 days */)
	// TODO(bburns) : examine response here.
	if err != nil {
		glog.Errorf("Error writing event: %s\n", err)
		if response != nil {
			glog.Infof("Response was: %v\n", *response)
		}
	}
	return err
}

func makeEnvironmentVariables(container *api.Container) []string {
	var result []string
	for _, value := range container.Env {
		result = append(result, fmt.Sprintf("%s=%s", value.Name, value.Value))
	}
	return result
}

func makeBinds(pod *Pod, container *api.Container, podVolumes volumeMap) []string {
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
		switch strings.ToUpper(port.Protocol) {
		case "UDP":
			protocol = "/udp"
		case "TCP":
			protocol = "/tcp"
		default:
			glog.Infof("Unknown protocol '%s': defaulting to TCP", port.Protocol)
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

func (kl *Kubelet) mountExternalVolumes(manifest *api.ContainerManifest) (volumeMap, error) {
	podVolumes := make(volumeMap)
	for _, vol := range manifest.Volumes {
		extVolume, err := volume.CreateVolumeBuilder(&vol, manifest.ID, kl.rootDirectory)
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
		glog.Errorf("Invalid handler: %v")
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
func (kl *Kubelet) runContainer(pod *Pod, container *api.Container, podVolumes volumeMap, netMode string) (id dockertools.DockerID, err error) {
	envVariables := makeEnvironmentVariables(container)
	binds := makeBinds(pod, container, podVolumes)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: dockertools.BuildDockerName(pod.Manifest.UUID, GetPodFullName(pod), container),
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
		handlerErr := kl.runHandler(GetPodFullName(pod), pod.Manifest.UUID, container, container.Lifecycle.PostStart)
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
	glog.Infof("Killing: %s", ID)
	err := kl.dockerClient.StopContainer(ID, 10)
	if len(name) == 0 {
		return err
	}
	podFullName, uuid, containerName, _ := dockertools.ParseDockerName(name)
	kl.LogEvent(&api.Event{
		Event: "STOP",
		Manifest: &api.ContainerManifest{
			//TODO: This should be reported using either the apiserver schema or the kubelet schema
			ID:   podFullName,
			UUID: uuid,
		},
		Container: &api.Container{
			Name: containerName,
		},
	})

	return err
}

const (
	networkContainerName  = "net"
	networkContainerImage = "kubernetes/pause:latest"
)

// createNetworkContainer starts the network container for a pod. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createNetworkContainer(pod *Pod) (dockertools.DockerID, error) {
	var ports []api.Port
	// Docker only exports ports from the network container.  Let's
	// collect all of the relevant ports and export them.
	for _, container := range pod.Manifest.Containers {
		ports = append(ports, container.Ports...)
	}
	container := &api.Container{
		Name:  networkContainerName,
		Image: networkContainerImage,
		Ports: ports,
	}
	kl.dockerPuller.Pull(networkContainerImage)
	return kl.runContainer(pod, container, nil, "")
}

// Delete all containers in a pod (except the network container) returns the number of containers deleted
// and an error if one occurs.
func (kl *Kubelet) deleteAllContainers(pod *Pod, podFullName string, dockerContainers dockertools.DockerContainers) (int, error) {
	count := 0
	errs := make(chan error, len(pod.Manifest.Containers))
	wg := sync.WaitGroup{}
	for _, container := range pod.Manifest.Containers {
		if dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, pod.Manifest.UUID, container.Name); found {
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

func (kl *Kubelet) syncPod(pod *Pod, dockerContainers dockertools.DockerContainers) error {
	podFullName := GetPodFullName(pod)
	uuid := pod.Manifest.UUID
	containersToKeep := make(map[dockertools.DockerID]empty)
	killedContainers := make(map[dockertools.DockerID]empty)

	// Make sure we have a network container
	var netID dockertools.DockerID
	if networkDockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, networkContainerName); found {
		netID = dockertools.DockerID(networkDockerContainer.ID)
	} else {
		glog.Infof("Network container doesn't exist, creating")
		count, err := kl.deleteAllContainers(pod, podFullName, dockerContainers)
		if err != nil {
			return err
		}
		dockerNetworkID, err := kl.createNetworkContainer(pod)
		if err != nil {
			glog.Errorf("Failed to introspect network container. (%v)  Skipping pod %s", err, podFullName)
			return err
		}
		netID = dockerNetworkID
		if count > 0 {
			// relist everything, otherwise we'll think we're ok
			dockerContainers, err = dockertools.GetKubeletDockerContainers(kl.dockerClient)
			if err != nil {
				glog.Errorf("Error listing containers %#v", dockerContainers)
				return err
			}
		}
	}
	containersToKeep[netID] = empty{}

	podVolumes, err := kl.mountExternalVolumes(&pod.Manifest)
	if err != nil {
		glog.Errorf("Unable to mount volumes for pod %s: (%v) Skipping pod.", podFullName, err)
		return err
	}

	podState := api.PodState{Manifest: api.ContainerManifest{UUID: uuid}}
	info, err := kl.GetPodInfo(podFullName, uuid)
	if err != nil {
		glog.Errorf("Unable to get pod with name %s and uuid %s info, health checks may be invalid.",
			podFullName, uuid)
	}
	netInfo, found := info[networkContainerName]
	if found && netInfo.NetworkSettings != nil {
		podState.PodIP = netInfo.NetworkSettings.IPAddress
	}

	for _, container := range pod.Manifest.Containers {
		expectedHash := dockertools.HashContainer(&container)
		if dockerContainer, found, hash := dockerContainers.FindPodContainer(podFullName, uuid, container.Name); found {
			containerID := dockertools.DockerID(dockerContainer.ID)
			glog.V(1).Infof("pod %s container %s exists as %v", podFullName, container.Name, containerID)

			// look for changes in the container.
			if hash == 0 || hash == expectedHash {
				// TODO: This should probably be separated out into a separate goroutine.
				healthy, err := kl.healthy(podFullName, podState, container, dockerContainer)
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
				glog.V(1).Infof("container hash changed %d vs %d.", hash, expectedHash)
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

		if len(recentContainers) > 0 && pod.Manifest.RestartPolicy.Always == nil {
			if pod.Manifest.RestartPolicy.Never != nil {
				glog.Infof("Already ran container with name %s--%s--%s, do nothing",
					podFullName, uuid, container.Name)
				continue
			}
			if pod.Manifest.RestartPolicy.OnFailure != nil {
				// Check the exit code of last run
				if recentContainers[0].State.ExitCode == 0 {
					glog.Infof("Already successfully ran container with name %s--%s--%s, do nothing",
						podFullName, uuid, container.Name)
					continue
				}
			}
		}

		glog.Infof("Container with name %s--%s--%s doesn't exist, creating %#v", podFullName, uuid, container.Name, container)
		if err := kl.dockerPuller.Pull(container.Image); err != nil {
			glog.Errorf("Failed to pull image %s: %v skipping pod %s container %s.", container.Image, err, podFullName, container.Name)
			continue
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
func getDesiredVolumes(pods []Pod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Manifest.Volumes {
			identifier := path.Join(pod.Manifest.ID, volume.Name)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
func (kl *Kubelet) reconcileVolumes(pods []Pod) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := volume.GetCurrentVolumes(kl.rootDirectory)
	for name, vol := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
			//TODO (jonesdl) We should somehow differentiate between volumes that are supposed
			//to be deleted and volumes that are leftover after a crash.
			glog.Infof("Orphaned volume %s found, tearing down volume", name)
			//TODO (jonesdl) This should not block other kubelet synchronization procedures
			err := vol.TearDown()
			if err != nil {
				glog.Infof("Could not tear down volume %s (%s)", name, err)
			}
		}
	}
	return nil
}

// SyncPods synchronizes the configured list of pods (desired state) with the host current state.
func (kl *Kubelet) SyncPods(pods []Pod) error {
	glog.Infof("Desired [%s]: %+v", kl.hostname, pods)
	var err error
	desiredContainers := make(map[podContainer]empty)

	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		glog.Errorf("Error listing containers %#v", dockerContainers)
		return err
	}

	// Check for any containers that need starting
	for i := range pods {
		pod := &pods[i]
		podFullName := GetPodFullName(pod)
		uuid := pod.Manifest.UUID

		// Add all containers (including net) to the map.
		desiredContainers[podContainer{podFullName, uuid, networkContainerName}] = empty{}
		for _, cont := range pod.Manifest.Containers {
			desiredContainers[podContainer{podFullName, uuid, cont.Name}] = empty{}
		}

		// Run the sync in an async manifest worker.
		kl.podWorkers.Run(podFullName, func() {
			err := kl.syncPod(pod, dockerContainers)
			if err != nil {
				glog.Errorf("Error syncing pod: %v skipping.", err)
			}
		})
	}

	// Kill any containers we don't need
	existingContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		glog.Errorf("Error listing containers: %v", err)
		return err
	}
	for _, container := range existingContainers {
		// Don't kill containers that are in the desired pods.
		podFullName, uuid, containerName, _ := dockertools.ParseDockerName(container.Names[0])
		if _, ok := desiredContainers[podContainer{podFullName, uuid, containerName}]; !ok {
			err = kl.killContainer(container)
			if err != nil {
				glog.Errorf("Error killing container: %v", err)
			}
		}
	}

	// Remove any orphaned volumes.
	kl.reconcileVolumes(pods)

	return err
}

// filterHostPortConflicts removes pods that conflict on Port.HostPort values
func filterHostPortConflicts(pods []Pod) []Pod {
	filtered := []Pod{}
	ports := map[int]bool{}
	extract := func(p *api.Port) int { return p.HostPort }
	for i := range pods {
		pod := &pods[i]
		if errs := validation.AccumulateUniquePorts(pod.Manifest.Containers, ports, extract); len(errs) != 0 {
			glog.Warningf("Pod %s has conflicting ports, ignoring: %v", GetPodFullName(pod), errs)
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
	var pods []Pod
	for {
		select {
		case u := <-updates:
			switch u.Op {
			case SET:
				glog.Infof("Containers changed [%s]", kl.hostname)
				pods = u.Pods
				pods = filterHostPortConflicts(pods)

			case UPDATE:
				//TODO: implement updates of containers
				glog.Infof("Containers updated, not implemented [%s]", kl.hostname)
				continue

			default:
				panic("syncLoop does not support incremental changes")
			}
		case <-time.After(kl.resyncInterval):
			if pods == nil {
				continue
			}
		}

		err := handler.SyncPods(pods)
		if err != nil {
			glog.Errorf("Couldn't sync containers : %v", err)
		}
	}
}

func getCadvisorContainerInfoRequest(req *info.ContainerInfoRequest) *info.ContainerInfoRequest {
	ret := &info.ContainerInfoRequest{
		NumStats: req.NumStats,
	}
	return ret
}

// This method takes a container's absolute path and returns the stats for the
// container.  The container's absolute path refers to its hierarchy in the
// cgroup file system. e.g. The root container, which represents the whole
// machine, has path "/"; all docker containers have path "/docker/<docker id>"
func (kl *Kubelet) statsFromContainerPath(containerPath string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	cinfo, err := kl.cadvisorClient.ContainerInfo(containerPath, getCadvisorContainerInfoRequest(req))
	if err != nil {
		return nil, err
	}
	return cinfo, nil
}

// GetKubeletContainerLogs returns logs from the container
// The second parameter of GetPodInfo and FindPodContainer methods represents pod UUID, which is allowed to be blank
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	_, err := kl.GetPodInfo(podFullName, "")
	if err == dockertools.ErrNoContainersInPod {
		return fmt.Errorf("Pod not found (%s)\n", podFullName)
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient)
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
	return dockertools.GetDockerPodInfo(kl.dockerClient, podFullName, uuid)
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName, uuid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	if kl.cadvisorClient == nil {
		return nil, nil
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		return nil, err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, containerName)
	if !found {
		return nil, errors.New("couldn't find container")
	}
	return kl.statsFromContainerPath(fmt.Sprintf("/docker/%s", dockerContainer.ID), req)
}

// GetRootInfo returns stats (from Cadvisor) of current machine (root container).
func (kl *Kubelet) GetRootInfo(req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return kl.statsFromContainerPath("/", req)
}

func (kl *Kubelet) GetMachineInfo() (*info.MachineInfo, error) {
	return kl.cadvisorClient.MachineInfo()
}

func (kl *Kubelet) healthy(podFullName string, currentState api.PodState, container api.Container, dockerContainer *docker.APIContainers) (health.Status, error) {
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
	return kl.healthChecker.HealthCheck(podFullName, currentState, container)
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
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		return nil, err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, container)
	if !found {
		return nil, fmt.Errorf("container not found (%s)", container)
	}
	return kl.runner.RunInContainer(dockerContainer.ID, cmd)
}
