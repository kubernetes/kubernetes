/*
Copyright 2015 Google Inc. All rights reserved.

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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

// Implements kubecontainer.ContainerRunner.
// TODO: Eventually DockerManager should implement kubecontainer.Runtime
// interface, and it should also add a cache to replace dockerCache.
type DockerManager struct {
	client   DockerInterface
	recorder record.EventRecorder
}

// Ensures DockerManager implements ConatinerRunner.
var _ kubecontainer.ContainerRunner = new(DockerManager)

func NewDockerManager(client DockerInterface, recorder record.EventRecorder) *DockerManager {
	return &DockerManager{client: client, recorder: recorder}
}

// GetKubeletDockerContainerLogs returns logs of a specific container. By
// default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
// TODO: Make 'RawTerminal' option  flagable.
func (self *DockerManager) GetKubeletDockerContainerLogs(containerID, tail string, follow bool, stdout, stderr io.Writer) (err error) {
	opts := docker.LogsOptions{
		Container:    containerID,
		Stdout:       true,
		Stderr:       true,
		OutputStream: stdout,
		ErrorStream:  stderr,
		Timestamps:   true,
		RawTerminal:  false,
		Follow:       follow,
	}

	if !follow {
		opts.Tail = tail
	}

	err = self.client.Logs(opts)
	return
}

var (
	// ErrNoContainersInPod is returned when there are no containers for a given pod
	ErrNoContainersInPod = errors.New("no containers exist for this pod")

	// ErrNoPodInfraContainerInPod is returned when there is no pod infra container for a given pod
	ErrNoPodInfraContainerInPod = errors.New("No pod infra container exists for this pod")

	// ErrContainerCannotRun is returned when a container is created, but cannot run properly
	ErrContainerCannotRun = errors.New("Container cannot run")
)

// Internal information kept for containers from inspection
type containerStatusResult struct {
	status api.ContainerStatus
	ip     string
	err    error
}

func (self *DockerManager) inspectContainer(dockerID, containerName, tPath string) *containerStatusResult {
	result := containerStatusResult{api.ContainerStatus{}, "", nil}

	inspectResult, err := self.client.InspectContainer(dockerID)

	if err != nil {
		result.err = err
		return &result
	}
	if inspectResult == nil {
		// Why did we not get an error?
		return &result
	}

	glog.V(3).Infof("Container inspect result: %+v", *inspectResult)
	result.status = api.ContainerStatus{
		Name:        containerName,
		Image:       inspectResult.Config.Image,
		ImageID:     DockerPrefix + inspectResult.Image,
		ContainerID: DockerPrefix + dockerID,
	}

	waiting := true
	if inspectResult.State.Running {
		result.status.State.Running = &api.ContainerStateRunning{
			StartedAt: util.NewTime(inspectResult.State.StartedAt),
		}
		if containerName == PodInfraContainerName && inspectResult.NetworkSettings != nil {
			result.ip = inspectResult.NetworkSettings.IPAddress
		}
		waiting = false
	} else if !inspectResult.State.FinishedAt.IsZero() {
		reason := ""
		// Note: An application might handle OOMKilled gracefully.
		// In that case, the container is oom killed, but the exit
		// code could be 0.
		if inspectResult.State.OOMKilled {
			reason = "OOM Killed"
		} else {
			reason = inspectResult.State.Error
		}
		result.status.State.Termination = &api.ContainerStateTerminated{
			ExitCode:   inspectResult.State.ExitCode,
			Reason:     reason,
			StartedAt:  util.NewTime(inspectResult.State.StartedAt),
			FinishedAt: util.NewTime(inspectResult.State.FinishedAt),
		}
		if tPath != "" {
			path, found := inspectResult.Volumes[tPath]
			if found {
				data, err := ioutil.ReadFile(path)
				if err != nil {
					glog.Errorf("Error on reading termination-log %s: %v", path, err)
				} else {
					result.status.State.Termination.Message = string(data)
				}
			}
		}
		waiting = false
	}

	if waiting {
		// TODO(dchen1107): Separate issue docker/docker#8294 was filed
		// TODO(dchen1107): Need to figure out why we are still waiting
		// Check any issue to run container
		result.status.State.Waiting = &api.ContainerStateWaiting{
			Reason: ErrContainerCannotRun.Error(),
		}
	}

	return &result
}

// GetPodStatus returns docker related status for all containers in the pod as
// well as the infrastructure container.
func (self *DockerManager) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID
	manifest := pod.Spec

	var podStatus api.PodStatus
	statuses := make(map[string]api.ContainerStatus)

	expectedContainers := make(map[string]api.Container)
	for _, container := range manifest.Containers {
		expectedContainers[container.Name] = container
	}
	expectedContainers[PodInfraContainerName] = api.Container{}

	containers, err := self.client.ListContainers(docker.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}

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
		terminationMessagePath := ""
		if !found {
			// TODO(dchen1107): should figure out why not continue here
			// continue
		} else {
			terminationMessagePath = c.TerminationMessagePath
		}
		// We assume docker return us a list of containers in time order
		if containerStatus, found := statuses[dockerContainerName]; found {
			containerStatus.RestartCount += 1
			statuses[dockerContainerName] = containerStatus
			continue
		}

		result := self.inspectContainer(value.ID, dockerContainerName, terminationMessagePath)
		if result.err != nil {
			return nil, result.err
		}

		// Add user container information
		if dockerContainerName == PodInfraContainerName &&
			result.status.State.Running != nil {
			// Found network container
			podStatus.PodIP = result.ip
		} else {
			statuses[dockerContainerName] = result.status
		}
	}

	if len(statuses) == 0 && podStatus.PodIP == "" {
		return nil, ErrNoContainersInPod
	}

	// Not all containers expected are created, check if there are
	// image related issues
	if len(statuses) < len(manifest.Containers) {
		var containerStatus api.ContainerStatus
		for _, container := range manifest.Containers {
			if _, found := statuses[container.Name]; found {
				continue
			}

			image := container.Image
			// Check image is ready on the node or not
			// TODO(dchen1107): docker/docker/issues/8365 to figure out if the image exists
			_, err := self.client.InspectImage(image)
			if err == nil {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: fmt.Sprintf("Image: %s is ready, container is creating", image),
				}
			} else if err == docker.ErrNoSuchImage {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: fmt.Sprintf("Image: %s is not ready on the node", image),
				}
			} else {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: "",
				}
			}

			statuses[container.Name] = containerStatus
		}
	}

	podStatus.ContainerStatuses = make([]api.ContainerStatus, 0)
	for _, status := range statuses {
		podStatus.ContainerStatuses = append(podStatus.ContainerStatuses, status)
	}

	return &podStatus, nil
}

func (self *DockerManager) GetRunningContainers(ids []string) ([]*docker.Container, error) {
	result := []*docker.Container{}
	if self.client == nil {
		return nil, fmt.Errorf("unexpected nil docker client.")
	}
	for ix := range ids {
		status, err := self.client.InspectContainer(ids[ix])
		if err != nil {
			return nil, err
		}
		if status != nil && status.State.Running {
			result = append(result, status)
		}
	}
	return result, nil
}

func (self *DockerManager) RunContainer(pod *api.Pod, container *api.Container, opts *kubecontainer.RunContainerOptions) (string, error) {
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	dockerName := KubeletContainerName{
		PodFullName:   kubecontainer.GetPodFullName(pod),
		PodUID:        pod.UID,
		ContainerName: container.Name,
	}
	exposedPorts, portBindings := makePortsAndBindings(container)

	// TODO(vmarmol): Handle better.
	// Cap hostname at 63 chars (specification is 64bytes which is 63 chars and the null terminating char).
	const hostnameMaxLen = 63
	containerHostname := pod.Name
	if len(containerHostname) > hostnameMaxLen {
		containerHostname = containerHostname[:hostnameMaxLen]
	}
	dockerOpts := docker.CreateContainerOptions{
		Name: BuildDockerName(dockerName, container),
		Config: &docker.Config{
			Env:          opts.Envs,
			ExposedPorts: exposedPorts,
			Hostname:     containerHostname,
			Image:        container.Image,
			Memory:       container.Resources.Limits.Memory().Value(),
			CPUShares:    milliCPUToShares(container.Resources.Limits.Cpu().MilliValue()),
			WorkingDir:   container.WorkingDir,
		},
	}

	setEntrypointAndCommand(container, &dockerOpts)

	glog.V(3).Infof("Container %v/%v/%v: setting entrypoint \"%v\" and command \"%v\"", pod.Namespace, pod.Name, container.Name, dockerOpts.Config.Entrypoint, dockerOpts.Config.Cmd)

	dockerContainer, err := self.client.CreateContainer(dockerOpts)
	if err != nil {
		if ref != nil {
			self.recorder.Eventf(ref, "failed", "Failed to create docker container with error: %v", err)
		}
		return "", err
	}

	if ref != nil {
		self.recorder.Eventf(ref, "created", "Created with docker id %v", dockerContainer.ID)
	}

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
			opts.Binds = append(opts.Binds, b)
		}
	}

	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = container.Privileged
	} else if container.Privileged {
		return "", fmt.Errorf("container requested privileged mode, but it is disallowed globally.")
	}

	capAdd, capDrop := makeCapabilites(container.Capabilities.Add, container.Capabilities.Drop)
	hc := &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        opts.Binds,
		NetworkMode:  opts.NetMode,
		IpcMode:      opts.IpcMode,
		Privileged:   privileged,
		CapAdd:       capAdd,
		CapDrop:      capDrop,
	}
	if len(opts.DNS) > 0 {
		hc.DNS = opts.DNS
	}
	if len(opts.DNSSearch) > 0 {
		hc.DNSSearch = opts.DNSSearch
	}

	if err = self.client.StartContainer(dockerContainer.ID, hc); err != nil {
		if ref != nil {
			self.recorder.Eventf(ref, "failed",
				"Failed to start with docker id %v with error: %v", dockerContainer.ID, err)
		}
		return "", err
	}
	if ref != nil {
		self.recorder.Eventf(ref, "started", "Started with docker id %v", dockerContainer.ID)
	}
	return dockerContainer.ID, nil
}

func setEntrypointAndCommand(container *api.Container, opts *docker.CreateContainerOptions) {
	if len(container.Command) != 0 {
		opts.Config.Entrypoint = container.Command
	}
	if len(container.Args) != 0 {
		opts.Config.Cmd = container.Args
	}
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
			glog.Warningf("Unknown protocol %q: defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			{
				HostPort: strconv.Itoa(exteriorPort),
				HostIP:   port.HostIP,
			},
		}
	}
	return exposedPorts, portBindings
}

func makeCapabilites(capAdd []api.CapabilityType, capDrop []api.CapabilityType) ([]string, []string) {
	var (
		addCaps  []string
		dropCaps []string
	)
	for _, cap := range capAdd {
		addCaps = append(addCaps, string(cap))
	}
	for _, cap := range capDrop {
		dropCaps = append(dropCaps, string(cap))
	}
	return addCaps, dropCaps
}
