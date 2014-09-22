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
	"errors"
	"fmt"
	"hash/adler32"
	"math/rand"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

// DockerContainerData is the structured representation of the JSON object returned by Docker inspect
type DockerContainerData struct {
	state struct {
		Running bool
	}
}

// DockerInterface is an abstract interface for testability.  It abstracts the interface of docker.Client.
type DockerInterface interface {
	ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
	Logs(opts docker.LogsOptions) error
}

// DockerID is an ID of docker container. It is a type to make it clear when we're working with docker container Ids
type DockerID string

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string) error
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client  DockerInterface
	keyring *dockerKeyring
}

// NewDockerPuller creates a new instance of the default implementation of DockerPuller.
func NewDockerPuller(client DockerInterface) DockerPuller {
	dp := dockerPuller{
		client:  client,
		keyring: newDockerKeyring(),
	}

	cfg, err := readDockerConfigFile()
	if err == nil {
		cfg.addToKeyring(dp.keyring)
	} else {
		glog.Errorf("Unable to parse docker config file: %v", err)
	}

	if dp.keyring.count() == 0 {
		glog.Infof("Continuing with empty docker keyring")
	}

	return dp
}

type dockerContainerCommandRunner struct{}

func (d *dockerContainerCommandRunner) getRunInContainerCommand(containerID string, cmd []string) (*exec.Cmd, error) {
	args := append([]string{"exec"}, cmd...)
	command := exec.Command("/usr/sbin/nsinit", args...)
	command.Dir = fmt.Sprintf("/var/lib/docker/execdriver/native/%s", containerID)
	return command, nil
}

// RunInContainer uses nsinit to run the command inside the container identified by containerID
func (d *dockerContainerCommandRunner) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	c, err := d.getRunInContainerCommand(containerID, cmd)
	if err != nil {
		return nil, err
	}
	return c.CombinedOutput()
}

// NewDockerContainerCommandRunner creates a ContainerCommandRunner which uses nsinit to run a command
// inside a container.
func NewDockerContainerCommandRunner() ContainerCommandRunner {
	return &dockerContainerCommandRunner{}
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

// DockerContainers is a map of containers
type DockerContainers map[DockerID]*docker.APIContainers

func (c DockerContainers) FindPodContainer(podFullName, uuid, containerName string) (*docker.APIContainers, bool, uint64) {
	for _, dockerContainer := range c {
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

// GetKubeletDockerContainers returns a map of docker containers that we manage. The map key is the docker container ID
func GetKubeletDockerContainers(client DockerInterface) (DockerContainers, error) {
	result := make(DockerContainers)
	containers, err := client.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return nil, err
	}
	for i := range containers {
		container := &containers[i]
		// Skip containers that we didn't create to allow users to manually
		// spin up their own containers if they want.
		if !strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"--") {
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

// ErrNoContainersInPod is returned when there are no containers for a given pod
var ErrNoContainersInPod = errors.New("no containers exist for this pod")

// GetDockerPodInfo returns docker info for all containers in the pod/manifest.
func GetDockerPodInfo(client DockerInterface, podFullName, uuid string) (api.PodInfo, error) {
	info := api.PodInfo{}

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
		// We assume docker return us a list of containers in time order
		if _, ok := info[dockerContainerName]; ok {
			continue
		}

		inspectResult, err := client.InspectContainer(value.ID)
		if err != nil {
			return nil, err
		}
		if inspectResult == nil {
			// Why did we not get an error?
			info[dockerContainerName] = docker.Container{}
		} else {
			info[dockerContainerName] = *inspectResult
		}
	}
	if len(info) == 0 {
		return nil, ErrNoContainersInPod
	}

	return info, nil
}

// Converts "-" to "_-_" and "_" to "___" so that we can use "--" to meaningfully separate parts of a docker name.
func escapeDash(in string) (out string) {
	out = strings.Replace(in, "_", "___", -1)
	out = strings.Replace(out, "-", "_-_", -1)
	return
}

// Reverses the transformation of escapeDash.
func unescapeDash(in string) (out string) {
	out = strings.Replace(in, "_-_", "-", -1)
	out = strings.Replace(out, "___", "_", -1)
	return
}

const containerNamePrefix = "k8s"

func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	fmt.Fprintf(hash, "%#v", *container)
	return uint64(hash.Sum32())
}

// Creates a name which can be reversed to identify both full pod name and container name.
func BuildDockerName(manifestUUID, podFullName string, container *api.Container) string {
	containerName := escapeDash(container.Name) + "." + strconv.FormatUint(HashContainer(container), 16)
	// Note, manifest.ID could be blank.
	if len(manifestUUID) == 0 {
		return fmt.Sprintf("%s--%s--%s--%08x",
			containerNamePrefix,
			containerName,
			escapeDash(podFullName),
			rand.Uint32())
	} else {
		return fmt.Sprintf("%s--%s--%s--%s--%08x",
			containerNamePrefix,
			containerName,
			escapeDash(podFullName),
			escapeDash(manifestUUID),
			rand.Uint32())
	}
}

// Upacks a container name, returning the pod full name and container name we would have used to
// construct the docker name. If the docker name isn't one we created, we may return empty strings.
func ParseDockerName(name string) (podFullName, uuid, containerName string, hash uint64) {
	// For some reason docker appears to be appending '/' to names.
	// If it's there, strip it.
	if name[0] == '/' {
		name = name[1:]
	}
	parts := strings.Split(name, "--")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		return
	}
	if len(parts) > 1 {
		pieces := strings.Split(parts[1], ".")
		containerName = unescapeDash(pieces[0])
		if len(pieces) > 1 {
			var err error
			hash, err = strconv.ParseUint(pieces[1], 16, 32)
			if err != nil {
				glog.Infof("invalid container hash: %s", pieces[1])
			}
		}
	}
	if len(parts) > 2 {
		podFullName = unescapeDash(parts[2])
	}
	if len(parts) > 4 {
		uuid = unescapeDash(parts[3])
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

func (dk *dockerKeyring) lookup(image string) (docker.AuthConfiguration, bool) {
	// range over the index as iterating over a map does not provide
	// a predictable ordering
	for _, k := range dk.index {
		if !strings.HasPrefix(image, k) {
			continue
		}

		return dk.creds[k], true
	}

	return docker.AuthConfiguration{}, false
}

func (dk dockerKeyring) count() int {
	return len(dk.creds)
}
