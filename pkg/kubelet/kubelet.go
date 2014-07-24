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
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
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
	dc DockerInterface,
	cc CadvisorInterface,
	ec tools.EtcdClient,
	rd string) *Kubelet {
	return &Kubelet{
		hostname:       hn,
		dockerClient:   dc,
		cadvisorClient: cc,
		etcdClient:     ec,
		rootDirectory:  rd,
		podWorkers:     newPodWorkers(),
	}
}

// NewIntegrationTestKubelet creates a new Kubelet for use in integration tests.
// TODO: add more integration tests, and expand parameter list as needed.
func NewIntegrationTestKubelet(hn string, dc DockerInterface) *Kubelet {
	return &Kubelet{
		hostname:     hn,
		dockerClient: dc,
		dockerPuller: &FakeDockerPuller{},
		podWorkers:   newPodWorkers(),
	}
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	hostname      string
	dockerClient  DockerInterface
	rootDirectory string
	podWorkers    podWorkers

	// Optional, no events will be sent without it
	etcdClient tools.EtcdClient
	// Optional, no statistics will be available if omitted
	cadvisorClient CadvisorInterface
	// Optional, defaults to simple implementaiton
	healthChecker health.HealthChecker
	// Optional, defaults to simple Docker implementation
	dockerPuller DockerPuller
	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.dockerPuller == nil {
		kl.dockerPuller = NewDockerPuller(kl.dockerClient)
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

func makeVolumesAndBinds(pod *Pod, container *api.Container, podVolumes volumeMap) (map[string]struct{}, []string) {
	volumes := map[string]struct{}{}
	binds := []string{}
	for _, volume := range container.VolumeMounts {
		var basePath string
		if vol, ok := podVolumes[volume.Name]; ok {
			// Host volumes are not Docker volumes and are directly mounted from the host.
			basePath = fmt.Sprintf("%s:%s", vol.GetPath(), volume.MountPath)
		} else if volume.MountType == "HOST" {
			// DEPRECATED: VolumeMount.MountType will be handled by the Volume struct.
			basePath = fmt.Sprintf("%s:%s", volume.MountPath, volume.MountPath)
		} else {
			// TODO(jonesdl) This clause should be deleted and an error should be thrown. The default
			// behavior is now supported by the EmptyDirectory type.
			volumes[volume.MountPath] = struct{}{}
			basePath = fmt.Sprintf("/exports/%s/%s:%s", GetPodFullName(pod), volume.Name, volume.MountPath)
		}
		if volume.ReadOnly {
			basePath += ":ro"
		}
		binds = append(binds, basePath)
	}
	return volumes, binds
}

func makePortsAndBindings(container *api.Container) (map[docker.Port]struct{}, map[docker.Port][]docker.PortBinding) {
	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range container.Ports {
		interiorPort := port.ContainerPort
		exteriorPort := port.HostPort
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
		extVolume, err := volume.CreateVolume(&vol, manifest.ID, kl.rootDirectory)
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

// Run a single container from a pod. Returns the docker container ID
func (kl *Kubelet) runContainer(pod *Pod, container *api.Container, podVolumes volumeMap, netMode string) (id DockerID, err error) {
	envVariables := makeEnvironmentVariables(container)
	volumes, binds := makeVolumesAndBinds(pod, container, podVolumes)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: buildDockerName(pod, container),
		Config: &docker.Config{
			Cmd:          container.Command,
			Env:          envVariables,
			ExposedPorts: exposedPorts,
			Hostname:     container.Name,
			Image:        container.Image,
			Memory:       uint64(container.Memory),
			CpuShares:    int64(milliCPUToShares(container.CPU)),
			Volumes:      volumes,
			WorkingDir:   container.WorkingDir,
		},
	}
	dockerContainer, err := kl.dockerClient.CreateContainer(opts)
	if err != nil {
		return "", err
	}
	err = kl.dockerClient.StartContainer(dockerContainer.ID, &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
	})
	return DockerID(dockerContainer.ID), err
}

// Kill a docker container
func (kl *Kubelet) killContainer(dockerContainer docker.APIContainers) error {
	glog.Infof("Killing: %s", dockerContainer.ID)
	err := kl.dockerClient.StopContainer(dockerContainer.ID, 10)
	podFullName, containerName := parseDockerName(dockerContainer.Names[0])
	kl.LogEvent(&api.Event{
		Event: "STOP",
		Manifest: &api.ContainerManifest{
			//TODO: This should be reported using either the apiserver schema or the kubelet schema
			ID: podFullName,
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
func (kl *Kubelet) createNetworkContainer(pod *Pod) (DockerID, error) {
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

type empty struct{}

func (kl *Kubelet) syncPod(pod *Pod, dockerContainers DockerContainers) error {
	podFullName := GetPodFullName(pod)
	containersToKeep := make(map[DockerID]empty)
	killedContainers := make(map[DockerID]empty)

	// Make sure we have a network container
	var netID DockerID
	if networkDockerContainer, found := dockerContainers.FindPodContainer(podFullName, networkContainerName); found {
		netID = DockerID(networkDockerContainer.ID)
	} else {
		glog.Infof("Network container doesn't exist, creating")
		dockerNetworkID, err := kl.createNetworkContainer(pod)
		if err != nil {
			glog.Errorf("Failed to introspect network container. (%v)  Skipping pod %s", err, podFullName)
			return err
		}
		netID = dockerNetworkID
	}
	containersToKeep[netID] = empty{}

	podVolumes, err := kl.mountExternalVolumes(&pod.Manifest)
	if err != nil {
		glog.Errorf("Unable to mount volumes for pod %s: (%v) Skipping pod.", podFullName, err)
		return err
	}

	for _, container := range pod.Manifest.Containers {
		if dockerContainer, found := dockerContainers.FindPodContainer(podFullName, container.Name); found {
			containerID := DockerID(dockerContainer.ID)
			glog.Infof("pod %s container %s exists as %v", podFullName, container.Name, containerID)
			glog.V(1).Infof("pod %s container %s exists as %v", podFullName, container.Name, containerID)

			// TODO: This should probably be separated out into a separate goroutine.
			healthy, err := kl.healthy(container, dockerContainer)
			if err != nil {
				glog.V(1).Infof("health check errored: %v", err)
				continue
			}
			if healthy == health.Healthy {
				containersToKeep[containerID] = empty{}
				continue
			}

			glog.V(1).Infof("pod %s container %s is unhealthy.", podFullName, container.Name, healthy)
			if err := kl.killContainer(*dockerContainer); err != nil {
				glog.V(1).Infof("Failed to kill container %s: %v", dockerContainer.ID, err)
				continue
			}
			killedContainers[containerID] = empty{}
		}

		glog.Infof("Container doesn't exist, creating %#v", container)
		if err := kl.dockerPuller.Pull(container.Image); err != nil {
			glog.Errorf("Failed to pull image: %v skipping pod %s container %s.", err, podFullName, container.Name)
			continue
		}
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
		curPodFullName, _ := parseDockerName(container.Names[0])
		if curPodFullName == podFullName {
			// Don't kill containers we want to keep or those we already killed.
			_, keep := containersToKeep[id]
			_, killed := killedContainers[id]
			if !keep && !killed {
				err = kl.killContainer(*container)
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
	containerName string
}

// SyncPods synchronizes the configured list of pods (desired state) with the host current state.
func (kl *Kubelet) SyncPods(pods []Pod) error {
	glog.Infof("Desired [%s]: %+v", kl.hostname, pods)
	var err error
	desiredContainers := make(map[podContainer]empty)

	dockerContainers, err := getKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		glog.Errorf("Error listing containers %#v", dockerContainers)
		return err
	}

	// Check for any containers that need starting
	for i := range pods {
		pod := &pods[i]
		podFullName := GetPodFullName(pod)

		// Add all containers (including net) to the map.
		desiredContainers[podContainer{podFullName, networkContainerName}] = empty{}
		for _, cont := range pod.Manifest.Containers {
			desiredContainers[podContainer{podFullName, cont.Name}] = empty{}
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
	existingContainers, err := getKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		glog.Errorf("Error listing containers: %v", err)
		return err
	}
	for _, container := range existingContainers {
		// Don't kill containers that are in the desired pods.
		podFullName, containerName := parseDockerName(container.Names[0])
		if _, ok := desiredContainers[podContainer{podFullName, containerName}]; !ok {
			err = kl.killContainer(*container)
			if err != nil {
				glog.Errorf("Error killing container: %v", err)
			}
		}
	}
	return err
}

// filterHostPortConflicts removes pods that conflict on Port.HostPort values
func filterHostPortConflicts(pods []Pod) []Pod {
	filtered := []Pod{}
	ports := map[int]bool{}
	extract := func(p *api.Port) int { return p.HostPort }
	for i := range pods {
		pod := &pods[i]
		if errs := api.AccumulateUniquePorts(pod.Manifest.Containers, ports, extract); len(errs) != 0 {
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
	for {
		var pods []Pod
		select {
		case u := <-updates:
			switch u.Op {
			case SET:
				glog.Infof("Containers changed [%s]", kl.hostname)
				pods = u.Pods

			case UPDATE:
				//TODO: implement updates of containers
				glog.Infof("Containers updated, not implemented [%s]", kl.hostname)
				continue

			default:
				panic("syncLoop does not support incremental changes")
			}
		}

		pods = filterHostPortConflicts(pods)

		err := handler.SyncPods(pods)
		if err != nil {
			glog.Errorf("Couldn't sync containers : %v", err)
		}
	}
}

func getCadvisorContainerInfoRequest(req *info.ContainerInfoRequest) *info.ContainerInfoRequest {
	ret := &info.ContainerInfoRequest{
		NumStats:               req.NumStats,
		CpuUsagePercentiles:    req.CpuUsagePercentiles,
		MemoryUsagePercentages: req.MemoryUsagePercentages,
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

// GetPodInfo returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodInfo(podFullName string) (api.PodInfo, error) {
	return getDockerPodInfo(kl.dockerClient, podFullName)
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	if kl.cadvisorClient == nil {
		return nil, nil
	}
	dockerContainers, err := getKubeletDockerContainers(kl.dockerClient)
	if err != nil {
		return nil, err
	}
	dockerContainer, found := dockerContainers.FindPodContainer(podFullName, containerName)
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

func (kl *Kubelet) healthy(container api.Container, dockerContainer *docker.APIContainers) (health.Status, error) {
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
	return kl.healthChecker.HealthCheck(container)
}

// Returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}
