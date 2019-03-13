// Copyright 2018 Google Inc. All Rights Reserved.
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

package mesos

import (
	"flag"
	"fmt"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager/watcher"
	"k8s.io/klog"
)

var MesosAgentAddress = flag.String("mesos_agent", "127.0.0.1:5051", "Mesos agent address")
var MesosAgentTimeout = flag.Duration("mesos_agent_timeout", 10*time.Second, "Mesos agent timeout")

// The namespace under which mesos aliases are unique.
const MesosNamespace = "mesos"

// Regexp that identifies mesos cgroups, containers started with
// --cgroup-parent have another prefix than 'mesos'
var mesosCgroupRegexp = regexp.MustCompile(`([a-z-0-9]{36})`)

// mesosFactory implements the interface ContainerHandlerFactory
type mesosFactory struct {
	machineInfoFactory info.MachineInfoFactory

	// Information about the cgroup subsystems.
	cgroupSubsystems libcontainer.CgroupSubsystems

	// Information about mounted filesystems.
	fsInfo fs.FsInfo

	includedMetrics map[container.MetricKind]struct{}

	client mesosAgentClient
}

func (self *mesosFactory) String() string {
	return MesosNamespace
}

func (self *mesosFactory) NewContainerHandler(name string, inHostNamespace bool) (container.ContainerHandler, error) {
	client, err := Client()
	if err != nil {
		return nil, err
	}

	return newMesosContainerHandler(
		name,
		&self.cgroupSubsystems,
		self.machineInfoFactory,
		self.fsInfo,
		self.includedMetrics,
		inHostNamespace,
		client,
	)
}

// ContainerNameToMesosId returns the Mesos ID from the full container name.
func ContainerNameToMesosId(name string) string {
	id := path.Base(name)

	if matches := mesosCgroupRegexp.FindStringSubmatch(id); matches != nil {
		return matches[1]
	}

	return id
}

// isContainerName returns true if the cgroup with associated name
// corresponds to a mesos container.
func isContainerName(name string) bool {
	// always ignore .mount cgroup even if associated with mesos and delegate to systemd
	if strings.HasSuffix(name, ".mount") {
		return false
	}
	return mesosCgroupRegexp.MatchString(path.Base(name))
}

// The mesos factory can handle any container.
func (self *mesosFactory) CanHandleAndAccept(name string) (handle bool, accept bool, err error) {
	// if the container is not associated with mesos, we can't handle it or accept it.
	if !isContainerName(name) {
		return false, false, nil
	}

	// Check if the container is known to mesos and it is active.
	id := ContainerNameToMesosId(name)

	_, err = self.client.ContainerInfo(id)
	if err != nil {
		return false, true, fmt.Errorf("error getting running container: %v", err)
	}

	return true, true, nil
}

func (self *mesosFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

func Register(
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	includedMetrics container.MetricSet,
) error {
	client, err := Client()

	if err != nil {
		return fmt.Errorf("unable to create mesos agent client: %v", err)
	}

	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems(includedMetrics)
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}

	klog.V(1).Infof("Registering mesos factory")
	factory := &mesosFactory{
		machineInfoFactory: machineInfoFactory,
		cgroupSubsystems:   cgroupSubsystems,
		fsInfo:             fsInfo,
		includedMetrics:    includedMetrics,
		client:             client,
	}
	container.RegisterContainerHandlerFactory(factory, []watcher.ContainerWatchSource{watcher.Raw})
	return nil
}
