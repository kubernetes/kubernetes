// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package docker

import (
	"flag"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"

	"github.com/docker/libcontainer/cgroups/systemd"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/info"
)

var ArgDockerEndpoint = flag.String("docker", "unix:///var/run/docker.sock", "docker endpoint")

type dockerFactory struct {
	machineInfoFactory info.MachineInfoFactory

	// Whether this system is using systemd.
	useSystemd bool

	client *docker.Client
}

func (self *dockerFactory) String() string {
	return "docker"
}

func (self *dockerFactory) NewContainerHandler(name string) (handler container.ContainerHandler, err error) {
	client, err := docker.NewClient(*ArgDockerEndpoint)
	if err != nil {
		return
	}
	handler, err = newDockerContainerHandler(
		client,
		name,
		self.machineInfoFactory,
		self.useSystemd,
	)
	return
}

// Docker handles all containers under /docker
// TODO(vishh): Change the CanHandle interface to be able to return errors.
func (self *dockerFactory) CanHandle(name string) bool {
	// In systemd systems the containers are: /system.slice/docker-{ID}
	if self.useSystemd {
		if !strings.HasPrefix(name, "/system.slice/docker-") {
			return false
		}
	} else if name == "/" {
		return false
	} else if name == "/docker" {
		// We need the docker driver to handle /docker. Otherwise the aggregation at the API level will break.
		return true
	} else if !strings.HasPrefix(name, "/docker/") {
		return false
	}
	// Check if the container is known to docker and it is active.
	_, id, err := libcontainer.SplitName(name)
	if err != nil {
		return false
	}
	ctnr, err := self.client.InspectContainer(id)
	// We assume that if Inspect fails then the container is not known to docker.
	// TODO(vishh): Detect lxc containers and avoid handling them.
	if err != nil || !ctnr.State.Running {
		return false
	}

	return true
}

func parseDockerVersion(full_version_string string) ([]int, error) {
	version_regexp_string := "(\\d+)\\.(\\d+)\\.(\\d+)"
	version_re := regexp.MustCompile(version_regexp_string)
	matches := version_re.FindAllStringSubmatch(full_version_string, -1)
	if len(matches) != 1 {
		return nil, fmt.Errorf("Version string \"%v\" doesn't match expected regular expression: \"%v\"", full_version_string, version_regexp_string)
	}
	version_string_array := matches[0][1:]
	version_array := make([]int, 3)
	for index, version_string := range version_string_array {
		version, err := strconv.Atoi(version_string)
		if err != nil {
			return nil, fmt.Errorf("Error while parsing \"%v\" in \"%v\"", version_string, full_version_string)
		}
		version_array[index] = version
	}
	return version_array, nil
}

// Register root container before running this function!
func Register(factory info.MachineInfoFactory) error {
	client, err := docker.NewClient(*ArgDockerEndpoint)
	if err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	if version, err := client.Version(); err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	} else {
		expected_version := []int{0, 11, 1}
		version_string := version.Get("Version")
		version, err := parseDockerVersion(version_string)
		if err != nil {
			return fmt.Errorf("Couldn't parse docker version: %v", err)
		}
		for index, number := range version {
			if number > expected_version[index] {
				break
			} else if number < expected_version[index] {
				return fmt.Errorf("cAdvisor requires docker version above %v but we have found version %v reported as \"%v\"", expected_version, version, version_string)
			}
		}
	}
	f := &dockerFactory{
		machineInfoFactory: factory,
		useSystemd:         systemd.UseSystemd(),
		client:             client,
	}
	if f.useSystemd {
		log.Printf("System is using systemd")
	}
	log.Printf("Registering Docker factory")
	container.RegisterContainerHandlerFactory(f)
	return nil
}
