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
	"sort"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
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
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
	Logs(opts docker.LogsOptions) error
	Version() (*docker.Env, error)
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
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
	client  DockerInterface
	keyring *dockerKeyring
}

type throttledDockerPuller struct {
	puller  dockerPuller
	limiter util.RateLimiter
}

// NewDockerPuller creates a new instance of the default implementation of DockerPuller.
func NewDockerPuller(client DockerInterface, qps float32, burst int) DockerPuller {
	dp := dockerPuller{
		client:  client,
		keyring: newDockerKeyring(),
	}

	cfg, err := readDockerConfigFile()
	if err == nil {
		cfg.addToKeyring(dp.keyring)
	} else if !os.IsNotExist(err) {
		glog.V(1).Infof("Unable to parse Docker config file: %v", err)
	}

	if dp.keyring.count() == 0 {
		glog.V(1).Infof("Continuing with empty Docker keyring")
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
	client DockerInterface
}

// The first version of docker that supports exec natively is 1.1.3
var dockerVersionWithExec = []uint{1, 1, 3}

// Returns the major and minor version numbers of docker server.
func (d *dockerContainerCommandRunner) getDockerServerVersion() ([]uint, error) {
	env, err := d.client.Version()
	if err != nil {
		return nil, fmt.Errorf("failed to get docker server version - %s", err)
	}
	version := []uint{}
	for _, entry := range *env {
		if strings.Contains(strings.ToLower(entry), "server version") {
			elems := strings.Split(strings.Split(entry, "=")[1], ".")
			for _, elem := range elems {
				val, err := strconv.ParseUint(elem, 10, 32)
				if err != nil {
					return nil, fmt.Errorf("failed to parse docker server version (%s) - %s", entry, err)
				}
				version = append(version, uint(val))
			}
			return version, nil
		}
	}
	return nil, fmt.Errorf("docker server version missing from server version output - %+v", env)
}

func (d *dockerContainerCommandRunner) nativeExecSupportExists() (bool, error) {
	version, err := d.getDockerServerVersion()
	if err != nil {
		return false, err
	}
	if len(dockerVersionWithExec) != len(version) {
		return false, fmt.Errorf("unexpected docker version format. Expecting %v format, got %v", dockerVersionWithExec, version)
	}
	for idx, val := range dockerVersionWithExec {
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
		return nil, fmt.Errorf("failed to run in container - Exec setup failed - %s", err)
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
		errChan <- d.client.StartExec(execObj.Id, startOpts)
	}()
	wrBuf.Flush()
	return buf.Bytes(), <-errChan
}

// NewDockerContainerCommandRunner creates a ContainerCommandRunner which uses nsinit to run a command
// inside a container.
func NewDockerContainerCommandRunner(client DockerInterface) ContainerCommandRunner {
	return &dockerContainerCommandRunner{client: client}
}

func (p dockerPuller) Pull(image string) error {
	image, tag := parseImageName(image)

	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = "latest"
	}

	opts := docker.PullImageOptions{
		Repository: image,
		Tag:        tag,
	}

	creds, ok := p.keyring.lookup(image)
	if !ok {
		glog.V(1).Infof("Pulling image %s without credentials", image)
	}

	return p.client.PullImage(opts, creds)
}

func (p throttledDockerPuller) Pull(image string) error {
	if p.limiter.CanAccept() {
		return p.puller.Pull(image)
	}
	return fmt.Errorf("pull QPS exceeded.")
}

func (p dockerPuller) IsImagePresent(name string) (bool, error) {
	image, _ := parseImageName(name)
	_, err := p.client.InspectImage(image)
	if err == nil {
		return true, nil
	}
	// This is super brittle, but its the best we got.
	// TODO: Land code in the docker client to use docker.Error here instead.
	if err.Error() == "no such image" {
		return false, nil
	}
	return false, err
}

func (p throttledDockerPuller) IsImagePresent(name string) (bool, error) {
	return p.puller.IsImagePresent(name)
}

// DockerContainers is a map of containers
type DockerContainers map[DockerID]*docker.APIContainers

func (c DockerContainers) FindPodContainer(podFullName, uuid, containerName string) (*docker.APIContainers, bool, uint64) {
	for _, dockerContainer := range c {
		// TODO(proppy): build the docker container name and do a map lookup instead?
		dockerManifestID, dockerUUID, dockerContainerName, hash := ParseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName &&
			(uuid == "" || dockerUUID == uuid) &&
			dockerContainerName == containerName {
			return dockerContainer, true, hash
		}
	}
	return nil, false, 0
}

// Note, this might return containers belong to a different Pod instance with the same name
func (c DockerContainers) FindContainersByPodFullName(podFullName string) map[string]*docker.APIContainers {
	containers := make(map[string]*docker.APIContainers)

	for _, dockerContainer := range c {
		dockerManifestID, _, dockerContainerName, _ := ParseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName {
			containers[dockerContainerName] = dockerContainer
		}
	}
	return containers
}

// GetKubeletDockerContainers takes client and boolean whether to list all container or just the running ones.
// Returns a map of docker containers that we manage. The map key is the docker container ID
func GetKubeletDockerContainers(client DockerInterface, allContainers bool) (DockerContainers, error) {
	result := make(DockerContainers)
	containers, err := client.ListContainers(docker.ListContainersOptions{All: allContainers})
	if err != nil {
		return nil, err
	}
	for i := range containers {
		container := &containers[i]
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
// and uuid given.
func GetRecentDockerContainersWithNameAndUUID(client DockerInterface, podFullName, uuid, containerName string) ([]*docker.Container, error) {
	var result []*docker.Container
	containers, err := client.ListContainers(docker.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}
	for _, dockerContainer := range containers {
		dockerPodName, dockerUUID, dockerContainerName, _ := ParseDockerName(dockerContainer.Names[0])
		if dockerPodName != podFullName {
			continue
		}
		if uuid != "" && dockerUUID != uuid {
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
func GetKubeletDockerContainerLogs(client DockerInterface, containerID, tail string, follow bool, stdout, stderr io.Writer) (err error) {
	opts := docker.LogsOptions{
		Container:    containerID,
		Stdout:       true,
		Stderr:       true,
		OutputStream: stdout,
		ErrorStream:  stderr,
		Timestamps:   true,
		RawTerminal:  true,
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

	// ErrNoNetworkContainerInPod is returned when there is no network container for a given pod
	ErrNoNetworkContainerInPod = errors.New("No network container exists for this pod")

	// ErrContainerCannotRun is returned when a container is created, but cannot run properly
	ErrContainerCannotRun = errors.New("Container cannot run")
)

func inspectContainer(client DockerInterface, dockerID, containerName, tPath string) (*api.ContainerStatus, error) {
	inspectResult, err := client.InspectContainer(dockerID)

	if err != nil {
		return nil, err
	}
	if inspectResult == nil {
		// Why did we not get an error?
		return &api.ContainerStatus{}, nil
	}

	glog.V(3).Infof("Container: %s [%s] inspect result %+v", *inspectResult)
	containerStatus := api.ContainerStatus{
		Image: inspectResult.Config.Image,
	}

	waiting := true
	if inspectResult.State.Running {
		containerStatus.State.Running = &api.ContainerStateRunning{
			StartedAt: inspectResult.State.StartedAt,
		}
		if containerName == "net" && inspectResult.NetworkSettings != nil {
			containerStatus.PodIP = inspectResult.NetworkSettings.IPAddress
		}
		waiting = false
	} else if !inspectResult.State.FinishedAt.IsZero() {
		// TODO(dchen1107): Integrate with event to provide a better reason
		containerStatus.State.Termination = &api.ContainerStateTerminated{
			ExitCode:   inspectResult.State.ExitCode,
			Reason:     "",
			StartedAt:  inspectResult.State.StartedAt,
			FinishedAt: inspectResult.State.FinishedAt,
		}
		if tPath != "" {
			path, found := inspectResult.Volumes[tPath]
			if found {
				data, err := ioutil.ReadFile(path)
				if err != nil {
					glog.Errorf("Error on reading termination-log %s(%v)", path, err)
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
func GetDockerPodInfo(client DockerInterface, manifest api.PodSpec, podFullName, uuid string) (api.PodInfo, error) {
	info := api.PodInfo{}
	expectedContainers := make(map[string]api.Container)
	for _, container := range manifest.Containers {
		expectedContainers[container.Name] = container
	}
	expectedContainers["net"] = api.Container{}

	containers, err := client.ListContainers(docker.ListContainersOptions{All: true})
	if err != nil {
		return nil, err
	}

	for _, value := range containers {
		dockerManifestID, dockerUUID, dockerContainerName, _ := ParseDockerName(value.Names[0])
		if dockerManifestID != podFullName {
			continue
		}
		if uuid != "" && dockerUUID != uuid {
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

	// First make sure we are not missing network container
	if _, found := info["net"]; !found {
		return nil, ErrNoNetworkContainerInPod
	}

	if len(info) < (len(manifest.Containers) + 1) {
		var containerStatus api.ContainerStatus
		// Not all containers expected are created, verify if there are
		// image related issues
		for _, container := range manifest.Containers {
			if _, found := info[container.Name]; found {
				continue
			}

			image := container.Image
			// Check image is ready on the node or not
			// TODO(dchen1107): docker/docker/issues/8365 to figure out if the image exists
			_, err := client.InspectImage(image)
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
					Reason: err.Error(),
				}
			}

			info[container.Name] = containerStatus
		}
	}

	return info, nil
}

const containerNamePrefix = "k8s"

func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	fmt.Fprintf(hash, "%#v", *container)
	return uint64(hash.Sum32())
}

// Creates a name which can be reversed to identify both full pod name and container name.
func BuildDockerName(manifestUUID, podFullName string, container *api.Container) string {
	containerName := container.Name + "." + strconv.FormatUint(HashContainer(container), 16)
	// Note, manifest.ID could be blank.
	if len(manifestUUID) == 0 {
		return fmt.Sprintf("%s_%s_%s_%08x",
			containerNamePrefix,
			containerName,
			podFullName,
			rand.Uint32())
	} else {
		return fmt.Sprintf("%s_%s_%s_%s_%08x",
			containerNamePrefix,
			containerName,
			podFullName,
			manifestUUID,
			rand.Uint32())
	}
}

// Unpacks a container name, returning the pod full name and container name we would have used to
// construct the docker name. If the docker name isn't the one we created, we may return empty strings.
func ParseDockerName(name string) (podFullName, uuid, containerName string, hash uint64) {
	// For some reason docker appears to be appending '/' to names.
	// If it's there, strip it.
	if name[0] == '/' {
		name = name[1:]
	}
	parts := strings.Split(name, "_")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		return
	}
	if len(parts) > 1 {
		pieces := strings.Split(parts[1], ".")
		containerName = pieces[0]
		if len(pieces) > 1 {
			var err error
			hash, err = strconv.ParseUint(pieces[1], 16, 32)
			if err != nil {
				glog.Warningf("invalid container hash: %s", pieces[1])
			}
		}
	}
	if len(parts) > 2 {
		podFullName = parts[2]
	}
	if len(parts) > 4 {
		uuid = parts[3]
	}
	return
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

type ContainerCommandRunner interface {
	RunInContainer(containerID string, cmd []string) ([]byte, error)
}

// dockerKeyring tracks a set of docker registry credentials, maintaining a
// reverse index across the registry endpoints. A registry endpoint is made
// up of a host (e.g. registry.example.com), but it may also contain a path
// (e.g. registry.example.com/foo) This index is important for two reasons:
// - registry endpoints may overlap, and when this happens we must find the
//   most specific match for a given image
// - iterating a map does not yield predictable results
type dockerKeyring struct {
	index []string
	creds map[string]docker.AuthConfiguration
}

func newDockerKeyring() *dockerKeyring {
	return &dockerKeyring{
		index: make([]string, 0),
		creds: make(map[string]docker.AuthConfiguration),
	}
}

func (dk *dockerKeyring) add(registry string, creds docker.AuthConfiguration) {
	dk.creds[registry] = creds

	dk.index = append(dk.index, registry)
	dk.reindex()
}

// reindex updates the index used to identify which credentials to use for
// a given image. The index is reverse-sorted so more specific paths are
// matched first. For example, if for the given image "quay.io/coreos/etcd",
// credentials for "quay.io/coreos" should match before "quay.io".
func (dk *dockerKeyring) reindex() {
	sort.Sort(sort.Reverse(sort.StringSlice(dk.index)))
}

const defaultRegistryHost = "index.docker.io/v1/"

func (dk *dockerKeyring) lookup(image string) (docker.AuthConfiguration, bool) {
	// range over the index as iterating over a map does not provide
	// a predictable ordering
	for _, k := range dk.index {
		if !strings.HasPrefix(image, k) {
			continue
		}

		return dk.creds[k], true
	}

	// use credentials for the default registry if provided
	if auth, ok := dk.creds[defaultRegistryHost]; ok {
		return auth, true
	}

	return docker.AuthConfiguration{}, false
}

func (dk dockerKeyring) count() int {
	return len(dk.creds)
}
