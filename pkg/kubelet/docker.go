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
	"math/rand"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/fsouza/go-dockerclient"
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

func (p dockerPuller) Pull(image string) error {
	image, tag := parseImageName(image)
	opts := docker.PullImageOptions{
		Repository: image,
		Tag:        tag,
	}
	return p.client.PullImage(opts, docker.AuthConfiguration{})
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

// Creates a name which can be reversed to identify both manifest id and container name.
func buildDockerName(manifest *api.ContainerManifest, container *api.Container) string {
	// Note, manifest.ID could be blank.
	return fmt.Sprintf("%s--%s--%s--%08x", containerNamePrefix, escapeDash(container.Name), escapeDash(manifest.ID), rand.Uint32())
}

// Upacks a container name, returning the manifest id and container name we would have used to
// construct the docker name. If the docker name isn't one we created, we may return empty strings.
func parseDockerName(name string) (manifestID, containerName string) {
	// For some reason docker appears to be appending '/' to names.
	// If its there, strip it.
	if name[0] == '/' {
		name = name[1:]
	}
	parts := strings.Split(name, "--")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		return
	}
	if len(parts) > 1 {
		containerName = unescapeDash(parts[1])
	}
	if len(parts) > 2 {
		manifestID = unescapeDash(parts[2])
	}
	return
}

// Parses image name including an tag and returns image name and tag
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
