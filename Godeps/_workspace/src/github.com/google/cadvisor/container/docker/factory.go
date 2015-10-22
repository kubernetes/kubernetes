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
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/docker/libcontainer/cgroups"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
)

var ArgDockerEndpoint = flag.String("docker", "unix:///var/run/docker.sock", "docker endpoint")

// The namespace under which Docker aliases are unique.
var DockerNamespace = "docker"

// Basepath to all container specific information that libcontainer stores.
var dockerRootDir = flag.String("docker_root", "/var/lib/docker", "Absolute path to the Docker state root directory (default: /var/lib/docker)")
var dockerRunDir = flag.String("docker_run", "/var/run/docker", "Absolute path to the Docker run directory (default: /var/run/docker)")

// Regexp that identifies docker cgroups, containers started with
// --cgroup-parent have another prefix than 'docker'
var dockerCgroupRegexp = regexp.MustCompile(`.+-([a-z0-9]{64})\.scope$`)

// TODO(vmarmol): Export run dir too for newer Dockers.
// Directory holding Docker container state information.
func DockerStateDir() string {
	return libcontainer.DockerStateDir(*dockerRootDir)
}

// Whether the system is using Systemd.
var useSystemd bool
var check = sync.Once{}

func UseSystemd() bool {
	check.Do(func() {
		useSystemd = false

		// Check for system.slice in systemd and cpu cgroup.
		for _, cgroupType := range []string{"name=systemd", "cpu"} {
			mnt, err := cgroups.FindCgroupMountpoint(cgroupType)
			if err == nil {
				// systemd presence does not mean systemd controls cgroups.
				// If system.slice cgroup exists, then systemd is taking control.
				// This breaks if user creates system.slice manually :)
				if utils.FileExists(path.Join(mnt, "system.slice")) {
					useSystemd = true
					break
				}
			}
		}
	})
	return useSystemd
}

func RootDir() string {
	return *dockerRootDir
}

type storageDriver string

const (
	devicemapperStorageDriver storageDriver = "devicemapper"
	aufsStorageDriver         storageDriver = "aufs"
)

type dockerFactory struct {
	machineInfoFactory info.MachineInfoFactory

	storageDriver storageDriver

	client *docker.Client

	// Information about the mounted cgroup subsystems.
	cgroupSubsystems libcontainer.CgroupSubsystems

	// Information about mounted filesystems.
	fsInfo fs.FsInfo
}

func (self *dockerFactory) String() string {
	return DockerNamespace
}

func (self *dockerFactory) NewContainerHandler(name string, inHostNamespace bool) (handler container.ContainerHandler, err error) {
	client, err := Client()
	if err != nil {
		return
	}
	handler, err = newDockerContainerHandler(
		client,
		name,
		self.machineInfoFactory,
		self.fsInfo,
		self.storageDriver,
		&self.cgroupSubsystems,
		inHostNamespace,
	)
	return
}

// Returns the Docker ID from the full container name.
func ContainerNameToDockerId(name string) string {
	id := path.Base(name)

	// Turn systemd cgroup name into Docker ID.
	if UseSystemd() {
		if matches := dockerCgroupRegexp.FindStringSubmatch(id); matches != nil {
			id = matches[1]
		}
	}

	return id
}

func isContainerName(name string) bool {
	if UseSystemd() {
		return dockerCgroupRegexp.MatchString(path.Base(name))
	}
	return true
}

// Docker handles all containers under /docker
func (self *dockerFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	// docker factory accepts all containers it can handle.
	canAccept := true

	if !isContainerName(name) {
		return false, canAccept, fmt.Errorf("invalid container name")
	}

	// Check if the container is known to docker and it is active.
	id := ContainerNameToDockerId(name)

	// We assume that if Inspect fails then the container is not known to docker.
	ctnr, err := self.client.InspectContainer(id)
	if err != nil || !ctnr.State.Running {
		return false, canAccept, fmt.Errorf("error inspecting container: %v", err)
	}

	return true, canAccept, nil
}

func (self *dockerFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

func parseDockerVersion(full_version_string string) ([]int, error) {
	version_regexp_string := "(\\d+)\\.(\\d+)\\.(\\d+)"
	version_re := regexp.MustCompile(version_regexp_string)
	matches := version_re.FindAllStringSubmatch(full_version_string, -1)
	if len(matches) != 1 {
		return nil, fmt.Errorf("version string \"%v\" doesn't match expected regular expression: \"%v\"", full_version_string, version_regexp_string)
	}
	version_string_array := matches[0][1:]
	version_array := make([]int, 3)
	for index, version_string := range version_string_array {
		version, err := strconv.Atoi(version_string)
		if err != nil {
			return nil, fmt.Errorf("error while parsing \"%v\" in \"%v\"", version_string, full_version_string)
		}
		version_array[index] = version
	}
	return version_array, nil
}

// Register root container before running this function!
func Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo) error {
	if UseSystemd() {
		glog.Infof("System is using systemd")
	}

	client, err := Client()
	if err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	if version, err := client.Version(); err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	} else {
		expected_version := []int{1, 0, 0}
		version_string := version.Get("Version")
		version, err := parseDockerVersion(version_string)
		if err != nil {
			return fmt.Errorf("couldn't parse docker version: %v", err)
		}
		for index, number := range version {
			if number > expected_version[index] {
				break
			} else if number < expected_version[index] {
				return fmt.Errorf("cAdvisor requires docker version %v or above but we have found version %v reported as \"%v\"", expected_version, version, version_string)
			}
		}
	}

	// Check that the libcontainer execdriver is used.
	information, err := DockerInfo()
	if err != nil {
		return fmt.Errorf("failed to detect Docker info: %v", err)
	}
	execDriver, ok := information["ExecutionDriver"]
	if !ok || !strings.HasPrefix(execDriver, "native") {
		return fmt.Errorf("docker found, but not using native exec driver")
	}

	sd, _ := information["Driver"]

	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems()
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}

	glog.Infof("Registering Docker factory")
	f := &dockerFactory{
		machineInfoFactory: factory,
		client:             client,
		storageDriver:      storageDriver(sd),
		cgroupSubsystems:   cgroupSubsystems,
		fsInfo:             fsInfo,
	}
	container.RegisterContainerHandlerFactory(f)
	return nil
}
