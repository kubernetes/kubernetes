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
	"regexp"
	"strconv"

	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info"
)

var ArgDockerEndpoint = flag.String("docker", "unix:///var/run/docker.sock", "docker endpoint")

type dockerFactory struct {
	machineInfoFactory info.MachineInfoFactory
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
	)
	return
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
func Register(factory info.MachineInfoFactory, paths ...string) error {
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
	}
	for _, p := range paths {
		if p != "/" && p != "/docker" {
			return fmt.Errorf("%v cannot be managed by docker", p)
		}
		container.RegisterContainerHandlerFactory(p, f)
	}
	return nil
}
