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

package dockertools

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash/adler32"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/leaky"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const (
	PodInfraContainerName = leaky.PodInfraContainerName
	DockerPrefix          = "docker://"
)

// DockerInterface is an abstract interface for testability.  It abstracts the interface of docker.Client.
type DockerInterface interface {
	ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
	RemoveContainer(opts docker.RemoveContainerOptions) error
	InspectImage(image string) (*docker.Image, error)
	ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error)
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
	RemoveImage(image string) error
	Logs(opts docker.LogsOptions) error
	Version() (*docker.Env, error)
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
}

// DockerRuntime implements the ContainerRuntime interface for docker.
type DockerRuntime struct {
	docker DockerInterface
}

// NewDockerRuntime creates a container.Runtime with a DockerInterface.
func NewDockerRuntime(docker DockerInterface) container.Runtime {
	return &DockerRuntime{docker: docker}
}

// ConvertAPIContainer converts a *docker.APIContainers to *container.Container.
func ConvertAPIContainer(dc *docker.APIContainers) *container.Container {
	if dc == nil {
		return nil
	}

	c := &container.Container{
		ID:         dc.ID,
		Image:      dc.Image,
		Command:    dc.Command,
		Created:    time.Unix(dc.Created, 0),
		Status:     dc.Status,
		SizeRw:     dc.SizeRw,
		SizeRootFs: dc.SizeRootFs,
		Names:      dc.Names,
	}

	// Copy Name.
	if len(dc.Names) == 0 {
		c.Name = ""
	} else {
		c.Name = dc.Names[0]
	}

	// Copy ports field too.
	c.Ports = make([]container.Port, len(dc.Ports))
	for i, p := range dc.Ports {
		c.Ports[i] = container.Port{
			PrivatePort: p.PrivatePort,
			PublicPort:  p.PublicPort,
			Type:        p.Type,
			IP:          p.IP,
		}
	}
	return c
}

// ConvertContainer converts a *docker.Container to *container.Container.
func ConvertContainer(dc *docker.Container) *container.Container {
	if dc == nil {
		return nil
	}

	var networkSettings *container.NetworkSettings
	var image string
	var state = container.State{
		Running:    dc.State.Running,
		Paused:     dc.State.Paused,
		OOMKilled:  dc.State.OOMKilled,
		Pid:        dc.State.Pid,
		ExitCode:   dc.State.ExitCode,
		Error:      dc.State.Error,
		StartedAt:  dc.State.StartedAt,
		FinishedAt: dc.State.FinishedAt,
	}
	if dc.NetworkSettings != nil {
		networkSettings = &container.NetworkSettings{
			IPAddress: dc.NetworkSettings.IPAddress,
		}
	}
	if dc.Config != nil {
		image = dc.Config.Image
	}

	return &container.Container{
		ID:              dc.ID,
		Name:            dc.Name,
		Names:           []string{dc.Name},
		Image:           image,
		ImageID:         dc.Image,
		Created:         dc.Created,
		State:           state,
		NetworkSettings: networkSettings,
		Volumes:         dc.Volumes,
	}
}

// ListContainers implements ContainerRuntime.ListContainers.
func (dr *DockerRuntime) ListContainers(options container.ListContainersOptions) ([]*container.Container, error) {
	var containers []*container.Container
	dc, err := dr.docker.ListContainers(docker.ListContainersOptions{All: options.All})
	for _, c := range dc {
		containers = append(containers, ConvertAPIContainer(&c))
	}
	return containers, err
}

// InspectContainer implements ContainerRuntime.InspectContainer.
func (dr *DockerRuntime) InspectContainer(id string) (*container.Container, error) {
	dc, err := dr.docker.InspectContainer(id)
	return ConvertContainer(dc), err
}

// CreateContainer implements ContainerRuntime.CreateContainer.
func (dr *DockerRuntime) CreateContainer(opts container.CreateContainerOptions) (*container.Container, error) {
	// Copy exposedPorts.
	exposedPorts := make(map[docker.Port]struct{})
	for port := range opts.ExposedPorts {
		exposedPorts[docker.Port(port)] = struct{}{}
	}

	dockerOptions := docker.CreateContainerOptions{
		Name: opts.Name,
		Config: &docker.Config{
			Cmd:          opts.Cmd,
			Env:          opts.Env,
			ExposedPorts: exposedPorts,
			Hostname:     opts.Hostname,
			Image:        opts.Image,
			Memory:       opts.Memory,
			CPUShares:    opts.CPUShares,
			WorkingDir:   opts.WorkingDir,
		},
	}

	dc, err := dr.docker.CreateContainer(dockerOptions)
	return ConvertContainer(dc), err
}

// StartContainer implements ContainerRuntime.StartContainer.
func (dr *DockerRuntime) StartContainer(id string, hc *container.HostConfig) error {
	dockerHostConfig := &docker.HostConfig{
		Binds:       hc.Binds,
		NetworkMode: hc.NetworkMode,
		IpcMode:     hc.IpcMode,
		Privileged:  hc.Privileged,
		CapAdd:      hc.CapAdd,
		CapDrop:     hc.CapDrop,
		DNS:         hc.DNS,
		DNSSearch:   hc.DNSSearch,
	}

	// Convert to docker.PortBinding.
	dockerHostConfig.PortBindings = make(map[docker.Port][]docker.PortBinding)
	for name, bindings := range hc.PortBindings {
		dockerPortBindings := make([]docker.PortBinding, len(bindings))
		for i := range bindings {
			dockerPortBindings[i] = docker.PortBinding{
				HostIP:   bindings[i].HostIP,
				HostPort: bindings[i].HostPort,
			}
		}
		dockerHostConfig.PortBindings[docker.Port(name)] = dockerPortBindings
	}
	return dr.docker.StartContainer(id, dockerHostConfig)
}

// StopContainer implements ContainerRuntime.StopContainer.
func (dr *DockerRuntime) StopContainer(id string, timeout uint) error {
	return dr.docker.StopContainer(id, timeout)
}

// RemoveContainer implements ContainerRuntime.RemoveContainer.
func (dr *DockerRuntime) RemoveContainer(opts container.RemoveContainerOptions) error {
	return dr.docker.RemoveContainer(docker.RemoveContainerOptions{ID: opts.ID})
}

// InspectImage implements ContainerRuntime.InspectImage.
func (dr *DockerRuntime) InspectImage(image string) (*container.Image, error) {
	img, err := dr.docker.InspectImage(image)
	if err != nil {
		if err == docker.ErrNoSuchImage {
			return nil, container.ErrNoSuchImage
		}
		return nil, err
	}
	if img == nil {
		return nil, nil
	}
	return &container.Image{ID: img.ID}, nil
}

// ListImage implements ContainerRuntime.ListImage.
func (dr *DockerRuntime) ListImages(opts container.ListImagesOptions) ([]*container.Image, error) {
	var imgs []*container.Image
	dockerImages, err := dr.docker.ListImages(docker.ListImagesOptions{All: opts.All})
	for i := range dockerImages {
		imgs = append(imgs, &container.Image{
			ID: dockerImages[i].ID,
		})
	}
	return imgs, err
}

// PullImage implements ContainerRuntime.PullImage.
func (dr *DockerRuntime) PullImage(opts container.PullImageOptions) error {
	dockerOpts := docker.PullImageOptions{
		Repository: opts.Repository,
		Tag:        opts.Tag,
	}
	return dr.docker.PullImage(dockerOpts, opts.DockerAuthConfig)
}

// RemoveImage implements ContainerRuntime.RemoveImage.
func (dr *DockerRuntime) RemoveImage(image string) error {
	return dr.docker.RemoveImage(image)
}

// Logs implements ContainerRuntime.Logs.
func (dr *DockerRuntime) Logs(opts container.LogsOptions) error {
	return dr.docker.Logs(docker.LogsOptions{
		Container:    opts.ID,
		OutputStream: opts.OutputStream,
		ErrorStream:  opts.ErrorStream,
		Follow:       opts.Follow,
		Stdout:       opts.Stdout,
		Stderr:       opts.Stderr,
		Timestamps:   opts.Timestamps,
		Tail:         opts.Tail,
		RawTerminal:  opts.RawTerminal,
	})
}

// Version implements ContainerRuntime.Version.
func (dr *DockerRuntime) Version() (*docker.Env, error) {
	return dr.docker.Version()
}

// CreateExec implements ContainerRuntime.CreateExec.
func (dr *DockerRuntime) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	return dr.docker.CreateExec(opts)
}

// StartExec implements ContainerRuntime.StartExec.
func (dr *DockerRuntime) StartExec(id string, opts docker.StartExecOptions) error {
	return dr.docker.StartExec(id, opts)
}

// DockerID is an ID of docker container. It is a type to make it clear when we're working with docker container Ids
type DockerID string

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string) error
	IsImagePresent(image string) (bool, error)
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client  container.Runtime
	keyring credentialprovider.DockerKeyring
}

type throttledDockerPuller struct {
	puller  dockerPuller
	limiter util.RateLimiter
}

// NewDockerPuller creates a new instance of the default implementation of DockerPuller.
func NewDockerPuller(client container.Runtime, qps float32, burst int) DockerPuller {
	dp := dockerPuller{
		client:  client,
		keyring: credentialprovider.NewDockerKeyring(),
	}

	if qps == 0.0 {
		return dp
	}
	return &throttledDockerPuller{
		puller:  dp,
		limiter: util.NewTokenBucketRateLimiter(qps, burst),
	}
}

type dockerContainerCommandRunner struct {
	client container.Runtime
}

// The first version of docker that supports exec natively is 1.3.0 == API 1.15
var dockerAPIVersionWithExec = []uint{1, 15}

// Returns the major and minor version numbers of docker server.
func (d *dockerContainerCommandRunner) GetDockerServerVersion() ([]uint, error) {
	env, err := d.client.Version()
	if err != nil {
		return nil, fmt.Errorf("failed to get docker server version - %v", err)
	}
	version := []uint{}
	for _, entry := range *env {
		if strings.Contains(strings.ToLower(entry), "apiversion") || strings.Contains(strings.ToLower(entry), "api version") {
			elems := strings.Split(strings.Split(entry, "=")[1], ".")
			for _, elem := range elems {
				val, err := strconv.ParseUint(elem, 10, 32)
				if err != nil {
					return nil, fmt.Errorf("failed to parse docker server version %q: %v", entry, err)
				}
				version = append(version, uint(val))
			}
			return version, nil
		}
	}
	return nil, fmt.Errorf("docker server version missing from server version output - %+v", env)
}

func (d *dockerContainerCommandRunner) nativeExecSupportExists() (bool, error) {
	version, err := d.GetDockerServerVersion()
	if err != nil {
		return false, err
	}
	if len(dockerAPIVersionWithExec) != len(version) {
		return false, fmt.Errorf("unexpected docker version format. Expecting %v format, got %v", dockerAPIVersionWithExec, version)
	}
	for idx, val := range dockerAPIVersionWithExec {
		if version[idx] < val {
			return false, nil
		}
	}
	return true, nil
}

func (d *dockerContainerCommandRunner) getRunInContainerCommand(containerID string, cmd []string) (*exec.Cmd, error) {
	args := append([]string{"exec"}, cmd...)
	command := exec.Command("/usr/sbin/nsinit", args...)
	command.Dir = fmt.Sprintf("/var/lib/docker/execdriver/native/%s", containerID)
	return command, nil
}

func (d *dockerContainerCommandRunner) runInContainerUsingNsinit(containerID string, cmd []string) ([]byte, error) {
	c, err := d.getRunInContainerCommand(containerID, cmd)
	if err != nil {
		return nil, err
	}
	return c.CombinedOutput()
}

// RunInContainer uses nsinit to run the command inside the container identified by containerID
func (d *dockerContainerCommandRunner) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	// If native exec support does not exist in the local docker daemon use nsinit.
	useNativeExec, err := d.nativeExecSupportExists()
	if err != nil {
		return nil, err
	}
	if !useNativeExec {
		return d.runInContainerUsingNsinit(containerID, cmd)
	}
	createOpts := docker.CreateExecOptions{
		Container:    containerID,
		Cmd:          cmd,
		AttachStdin:  false,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
	}
	execObj, err := d.client.CreateExec(createOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to run in container - Exec setup failed - %v", err)
	}
	var buf bytes.Buffer
	wrBuf := bufio.NewWriter(&buf)
	startOpts := docker.StartExecOptions{
		Detach:       false,
		Tty:          false,
		OutputStream: wrBuf,
		ErrorStream:  wrBuf,
		RawTerminal:  false,
	}
	errChan := make(chan error, 1)
	go func() {
		errChan <- d.client.StartExec(execObj.ID, startOpts)
	}()
	wrBuf.Flush()
	return buf.Bytes(), <-errChan
}

// ExecInContainer uses nsenter to run the command inside the container identified by containerID.
//
// TODO:
//  - match cgroups of container
//  - should we support `docker exec`?
//  - should we support nsenter in a container, running with elevated privs and --pid=host?
func (d *dockerContainerCommandRunner) ExecInContainer(containerId string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	container, err := d.client.InspectContainer(containerId)
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container)
	}

	containerPid := container.State.Pid

	// TODO what if the container doesn't have `env`???
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-m", "-i", "-u", "-n", "-p", "--", "env", "-i"}
	args = append(args, fmt.Sprintf("HOSTNAME=%s", container.Config.Hostname))
	args = append(args, container.Config.Env...)
	args = append(args, cmd...)
	glog.Infof("ARGS %#v", args)
	command := exec.Command("nsenter", args...)
	// TODO use exec.LookPath
	if tty {
		p, err := StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		if stdin != nil {
			go io.Copy(p, stdin)
		}

		if stdout != nil {
			go io.Copy(stdout, p)
		}

		return command.Wait()
	} else {
		cp := func(dst io.WriteCloser, src io.Reader, closeDst bool) {
			defer func() {
				if closeDst {
					dst.Close()
				}
			}()
			io.Copy(dst, src)
		}
		if stdin != nil {
			inPipe, err := command.StdinPipe()
			if err != nil {
				return err
			}
			go func() {
				cp(inPipe, stdin, false)
				inPipe.Close()
			}()
		}

		if stdout != nil {
			outPipe, err := command.StdoutPipe()
			if err != nil {
				return err
			}
			go cp(stdout, outPipe, true)
		}

		if stderr != nil {
			errPipe, err := command.StderrPipe()
			if err != nil {
				return err
			}
			go cp(stderr, errPipe, true)
		}

		return command.Run()
	}
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
func (d *dockerContainerCommandRunner) PortForward(podInfraContainerID string, port uint16, stream io.ReadWriteCloser) error {
	container, err := d.client.InspectContainer(podInfraContainerID)
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container)
	}

	containerPid := container.State.Pid
	// TODO use exec.LookPath for socat / what if the host doesn't have it???
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-n", "socat", "-", fmt.Sprintf("TCP4:localhost:%d", port)}
	// TODO use exec.LookPath
	command := exec.Command("nsenter", args...)
	in, err := command.StdinPipe()
	if err != nil {
		return err
	}
	out, err := command.StdoutPipe()
	if err != nil {
		return err
	}
	go io.Copy(in, stream)
	go io.Copy(stream, out)
	return command.Run()
}

// NewDockerContainerCommandRunner creates a ContainerCommandRunner which uses nsinit to run a command
// inside a container.
func NewDockerContainerCommandRunner(client container.Runtime) ContainerCommandRunner {
	return &dockerContainerCommandRunner{client: client}
}

func (p dockerPuller) Pull(image string) error {
	image, tag := parseImageName(image)

	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = "latest"
	}

	auth, ok := p.keyring.Lookup(image)
	if !ok {
		glog.V(1).Infof("Pulling image %s without credentials", image)
	}

	opts := container.PullImageOptions{
		Repository:       image,
		Tag:              tag,
		DockerAuthConfig: auth,
	}

	err := p.client.PullImage(opts)
	// If there was no error, or we had credentials, just return the error.
	if err == nil || ok {
		return err
	}
	// Image spec: [<registry>/]<repository>/<image>[:<version] so we count '/'
	explicitRegistry := (strings.Count(image, "/") == 2)
	glog.Errorf("Foo: %s", explicitRegistry)
	// Hack, look for a private registry, and decorate the error with the lack of
	// credentials.  This is heuristic, and really probably could be done better
	// by talking to the registry API directly from the kubelet here.
	if explicitRegistry {
		return fmt.Errorf("image pull failed for %s, this may be because there are no credentials on this request.  details: (%v)", image, err)
	}
	return err
}

func (p throttledDockerPuller) Pull(image string) error {
	if p.limiter.CanAccept() {
		return p.puller.Pull(image)
	}
	return fmt.Errorf("pull QPS exceeded.")
}

func (p dockerPuller) IsImagePresent(image string) (bool, error) {
	_, err := p.client.InspectImage(image)
	if err == nil {
		return true, nil
	}
	// This is super brittle, but its the best we got.
	// TODO(yifan): Merge DockerPuller into container.Runtime later?
	if err == container.ErrNoSuchImage {
		return false, nil
	}
	return false, err
}

// RequireLatestImage returns if the user wants the latest image
func RequireLatestImage(name string) bool {
	_, tag := parseImageName(name)

	if tag == "latest" {
		return true
	}
	return false
}

func (p throttledDockerPuller) IsImagePresent(name string) (bool, error) {
	return p.puller.IsImagePresent(name)
}

// DockerContainers is a map of containers
type DockerContainers map[DockerID]*container.Container

func (c DockerContainers) FindPodContainer(podFullName string, uid types.UID, containerName string) (*container.Container, bool, uint64) {
	for _, dockerContainer := range c {
		if len(dockerContainer.Names) == 0 {
			continue
		}
		// TODO(proppy): build the docker container name and do a map lookup instead?
		dockerManifestID, dockerUUID, dockerContainerName, hash := ParseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName &&
			(uid == "" || dockerUUID == uid) &&
			dockerContainerName == containerName {
			return dockerContainer, true, hash
		}
	}
	return nil, false, 0
}

// Note, this might return containers belong to a different Pod instance with the same name
func (c DockerContainers) FindContainersByPodFullName(podFullName string) map[string]*container.Container {
	containers := make(map[string]*container.Container)

	for _, dockerContainer := range c {
		if len(dockerContainer.Names) == 0 {
			continue
		}
		dockerManifestID, _, dockerContainerName, _ := ParseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName {
			containers[dockerContainerName] = dockerContainer
		}
	}
	return containers
}

// GetKubeletDockerContainers takes client and boolean whether to list all container or just the running ones.
// Returns a map of docker containers that we manage. The map key is the docker container ID
func GetKubeletDockerContainers(client container.Runtime, allContainers bool) (DockerContainers, error) {
	result := make(DockerContainers)
	containers, err := client.ListContainers(container.ListContainersOptions{All: allContainers})
	if err != nil {
		return nil, err
	}
	for _, container := range containers {
		if len(container.Names) == 0 {
			continue
		}
		// Skip containers that we didn't create to allow users to manually
		// spin up their own containers if they want.
		// TODO(dchen1107): Remove the old separator "--" by end of Oct
		if !strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"_") &&
			!strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"--") {
			glog.V(3).Infof("Docker Container: %s is not managed by kubelet.", container.Names[0])
			continue
		}
		result[DockerID(container.ID)] = container
	}
	return result, nil
}

// GetRecentDockerContainersWithNameAndUUID returns a list of dead docker containers which matches the name
// and uid given.
func GetRecentDockerContainersWithNameAndUUID(client container.Runtime, podFullName string, uid types.UID, containerName string) ([]*container.Container, error) {
	var result []*container.Container
	containers, err := client.ListContainers(container.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}
	for _, dockerContainer := range containers {
		if len(dockerContainer.Names) == 0 {
			continue
		}
		dockerPodName, dockerUUID, dockerContainerName, _ := ParseDockerName(dockerContainer.Names[0])
		if dockerPodName != podFullName {
			continue
		}
		if uid != "" && dockerUUID != uid {
			continue
		}
		if dockerContainerName != containerName {
			continue
		}
		inspectResult, _ := client.InspectContainer(dockerContainer.ID)
		if inspectResult != nil && !inspectResult.State.Running && !inspectResult.State.Paused {
			result = append(result, inspectResult)
		}
	}
	return result, nil
}

// GetKubeletDockerContainerLogs returns logs of specific container
// By default the function will return snapshot of the container log
// Log streaming is possible if 'follow' param is set to true
// Log tailing is possible when number of tailed lines are set and only if 'follow' is false
// TODO: Make 'RawTerminal' option  flagable.
func GetKubeletDockerContainerLogs(client container.Runtime, containerID, tail string, follow bool, stdout, stderr io.Writer) (err error) {
	opts := container.LogsOptions{
		ID:           containerID,
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

	err = client.Logs(opts)
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

func inspectContainer(client container.Runtime, dockerID, containerName, tPath string) (*api.ContainerStatus, error) {
	inspectResult, err := client.InspectContainer(dockerID)

	if err != nil {
		return nil, err
	}
	if inspectResult == nil {
		// Why did we not get an error?
		return &api.ContainerStatus{}, nil
	}

	glog.V(3).Infof("Container inspect result: %+v", *inspectResult)
	containerStatus := api.ContainerStatus{
		Image:       inspectResult.Image,
		ImageID:     DockerPrefix + inspectResult.ImageID,
		ContainerID: DockerPrefix + dockerID,
	}

	waiting := true
	if inspectResult.State.Running {
		containerStatus.State.Running = &api.ContainerStateRunning{
			StartedAt: util.NewTime(inspectResult.State.StartedAt),
		}
		if containerName == PodInfraContainerName && inspectResult.NetworkSettings != nil {
			containerStatus.PodIP = inspectResult.NetworkSettings.IPAddress
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
		containerStatus.State.Termination = &api.ContainerStateTerminated{
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
					containerStatus.State.Termination.Message = string(data)
				}
			}
		}
		waiting = false
	}

	if waiting {
		// TODO(dchen1107): Separate issue docker/docker#8294 was filed
		// TODO(dchen1107): Need to figure out why we are still waiting
		// Check any issue to run container
		containerStatus.State.Waiting = &api.ContainerStateWaiting{
			Reason: ErrContainerCannotRun.Error(),
		}
	}

	return &containerStatus, nil
}

// GetDockerPodInfo returns docker info for all containers in the pod/manifest.
func GetDockerPodInfo(client container.Runtime, manifest api.PodSpec, podFullName string, uid types.UID) (api.PodInfo, error) {
	info := api.PodInfo{}
	expectedContainers := make(map[string]api.Container)
	for _, container := range manifest.Containers {
		expectedContainers[container.Name] = container
	}
	expectedContainers[PodInfraContainerName] = api.Container{}

	containers, err := client.ListContainers(container.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}

	for _, value := range containers {
		if len(value.Names) == 0 {
			continue
		}
		dockerManifestID, dockerUUID, dockerContainerName, _ := ParseDockerName(value.Names[0])
		if dockerManifestID != podFullName {
			continue
		}
		if uid != "" && dockerUUID != uid {
			continue
		}
		c, found := expectedContainers[dockerContainerName]
		terminationMessagePath := ""
		if !found {
			// TODO(dchen1107): should figure out why not continue here
			// continue
		} else {
			terminationMessagePath = c.TerminationMessagePath
		}
		// We assume docker return us a list of containers in time order
		if containerStatus, found := info[dockerContainerName]; found {
			containerStatus.RestartCount += 1
			info[dockerContainerName] = containerStatus
			continue
		}

		containerStatus, err := inspectContainer(client, value.ID, dockerContainerName, terminationMessagePath)
		if err != nil {
			return nil, err
		}
		info[dockerContainerName] = *containerStatus
	}

	if len(info) == 0 {
		return nil, ErrNoContainersInPod
	}

	// First make sure we are not missing pod infra container
	if _, found := info[PodInfraContainerName]; !found {
		return nil, ErrNoPodInfraContainerInPod
	}

	if len(info) < (len(manifest.Containers) + 1) {
		var containerStatus api.ContainerStatus
		// Not all containers expected are created, check if there are
		// image related issues
		for _, ctnr := range manifest.Containers {
			if _, found := info[ctnr.Name]; found {
				continue
			}

			image := ctnr.Image
			// Check image is ready on the node or not
			// TODO(dchen1107): docker/docker/issues/8365 to figure out if the image exists
			_, err := client.InspectImage(image)
			if err == nil {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: fmt.Sprintf("Image: %s is ready, container is creating", image),
				}
			} else if err == container.ErrNoSuchImage {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: fmt.Sprintf("Image: %s is not ready on the node", image),
				}
			} else {
				containerStatus.State.Waiting = &api.ContainerStateWaiting{
					Reason: err.Error(),
				}
			}

			info[ctnr.Name] = containerStatus
		}
	}

	return info, nil
}

const containerNamePrefix = "k8s"

func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	util.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}

// Creates a name which can be reversed to identify both full pod name and container name.
func BuildDockerName(podUID types.UID, podFullName string, container *api.Container) string {
	containerName := container.Name + "." + strconv.FormatUint(HashContainer(container), 16)
	return fmt.Sprintf("%s_%s_%s_%s_%08x",
		containerNamePrefix,
		containerName,
		podFullName,
		podUID,
		rand.Uint32())
}

// TODO(vmarmol): This should probably return an error.
// Unpacks a container name, returning the pod full name and container name we would have used to
// construct the docker name. If the docker name isn't the one we created, we may return empty strings.
func ParseDockerName(name string) (podFullName string, podUID types.UID, containerName string, hash uint64) {
	// For some reason docker appears to be appending '/' to names.
	// If it's there, strip it.
	if name[0] == '/' {
		name = name[1:]
	}
	parts := strings.Split(name, "_")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		return
	}
	if len(parts) < 5 {
		// We have at least 5 fields.  We may have more in the future.
		// Anything with less fields than this is not something we can
		// manage.
		glog.Warningf("found a container with the %q prefix, but too few fields (%d): ", containerNamePrefix, len(parts), name)
		return
	}

	// Container name.
	nameParts := strings.Split(parts[1], ".")
	containerName = nameParts[0]
	if len(nameParts) > 1 {
		var err error
		hash, err = strconv.ParseUint(nameParts[1], 16, 32)
		if err != nil {
			glog.Warningf("invalid container hash: %s", nameParts[1])
		}
	}

	// Pod fullname.
	podFullName = parts[2]

	// Pod UID.
	podUID = types.UID(parts[3])

	return
}

func GetRunningContainers(client container.Runtime, ids []string) ([]*container.Container, error) {
	var result []*container.Container
	if client == nil {
		return nil, fmt.Errorf("unexpected nil docker client.")
	}
	for ix := range ids {
		status, err := client.InspectContainer(ids[ix])
		if err != nil {
			return nil, err
		}
		if status != nil && status.State.Running {
			result = append(result, status)
		}
	}
	return result, nil
}

// Parses image name including a tag and returns image name and tag.
// TODO: Future Docker versions can parse the tag on daemon side, see
// https://github.com/dotcloud/docker/issues/6876
// So this can be deprecated at some point.
func parseImageName(image string) (string, string) {
	tag := ""
	parts := strings.SplitN(image, "/", 2)
	repo := ""
	if len(parts) == 2 {
		repo = parts[0]
		image = parts[1]
	}
	parts = strings.SplitN(image, ":", 2)
	if len(parts) == 2 {
		image = parts[0]
		tag = parts[1]
	}
	if repo != "" {
		image = fmt.Sprintf("%s/%s", repo, image)
	}
	return image, tag
}

// Get a docker endpoint, either from the string passed in, or $DOCKER_HOST environment variables
func getDockerEndpoint(dockerEndpoint string) string {
	var endpoint string
	if len(dockerEndpoint) > 0 {
		endpoint = dockerEndpoint
	} else if len(os.Getenv("DOCKER_HOST")) > 0 {
		endpoint = os.Getenv("DOCKER_HOST")
	} else {
		endpoint = "unix:///var/run/docker.sock"
	}
	glog.Infof("Connecting to docker on %s", endpoint)

	return endpoint
}

// ConnectToDockerOrDie creates a DockerRuntime.
func ConnectToDockerOrDie(dockerEndpoint string) container.Runtime {
	if dockerEndpoint == "fake://" {
		return &DockerRuntime{
			docker: &FakeDockerClient{
				VersionInfo: []string{"apiVersion=1.16"},
			},
		}
	}
	client, err := docker.NewClient(getDockerEndpoint(dockerEndpoint))
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}
	return &DockerRuntime{docker: client}
}

type ContainerCommandRunner interface {
	RunInContainer(containerID string, cmd []string) ([]byte, error)
	GetDockerServerVersion() ([]uint, error)
	ExecInContainer(containerID string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	PortForward(podInfraContainerID string, port uint16, stream io.ReadWriteCloser) error
}
