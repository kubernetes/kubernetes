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
	"errors"
	"fmt"
	"hash/adler32"
	"math/rand"
	"os/exec"
	"strconv"
	"strings"

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
}

// DockerID is an ID of docker container. It is a type to make it clear when we're working with docker container Ids
type DockerID string

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string) error
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client DockerInterface
}

// NewDockerPuller creates a new instance of the default implementation of DockerPuller.
func NewDockerPuller(client DockerInterface) DockerPuller {
	return dockerPuller{
		client: client,
	}
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
	return p.client.PullImage(opts, docker.AuthConfiguration{})
}

// DockerContainers is a map of containers
type DockerContainers map[DockerID]*docker.APIContainers

func (c DockerContainers) FindPodContainer(podFullName, containerName string) (*docker.APIContainers, bool, uint64) {
	for _, dockerContainer := range c {
		dockerManifestID, dockerContainerName, hash := parseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName && dockerContainerName == containerName {
			return dockerContainer, true, hash
		}
	}
	return nil, false, 0
}

func (c DockerContainers) FindContainersByPodFullName(podFullName string) map[string]*docker.APIContainers {
	containers := make(map[string]*docker.APIContainers)

	for _, dockerContainer := range c {
		dockerManifestID, dockerContainerName, _ := parseDockerName(dockerContainer.Names[0])
		if dockerManifestID == podFullName {
			containers[dockerContainerName] = dockerContainer
		}
	}
	return containers
}

// GetKubeletDockerContainers returns a map of docker containers that we manage. The map key is the docker container ID
func getKubeletDockerContainers(client DockerInterface) (DockerContainers, error) {
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

// ErrNoContainersInPod is returned when there are no running containers for a given pod
var ErrNoContainersInPod = errors.New("no containers exist for this pod")

// GetDockerPodInfo returns docker info for all containers in the pod/manifest.
func getDockerPodInfo(client DockerInterface, podFullName string) (api.PodInfo, error) {
	info := api.PodInfo{}

	containers, err := client.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return nil, err
	}

	for _, value := range containers {
		dockerManifestID, dockerContainerName, _ := parseDockerName(value.Names[0])
		if dockerManifestID != podFullName {
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

func hashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	fmt.Fprintf(hash, "%#v", *container)
	return uint64(hash.Sum32())
}

// Creates a name which can be reversed to identify both full pod name and container name.
func buildDockerName(pod *Pod, container *api.Container) string {
	containerName := escapeDash(container.Name) + "." + strconv.FormatUint(hashContainer(container), 16)
	// Note, manifest.ID could be blank.
	return fmt.Sprintf("%s--%s--%s--%08x", containerNamePrefix, containerName, escapeDash(GetPodFullName(pod)), rand.Uint32())
}

// Upacks a container name, returning the pod full name and container name we would have used to
// construct the docker name. If the docker name isn't one we created, we may return empty strings.
func parseDockerName(name string) (podFullName, containerName string, hash uint64) {
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
	return
}

// Parses image name including a tag and returns image name and tag.
// TODO: Future Docker versions can parse the tag on daemon side, see:
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
