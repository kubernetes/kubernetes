// +build linux

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package systemd

import (
	"fmt"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm/bootstrap"
	"k8s.io/kubernetes/pkg/util/mount"
)

// ensure we implement the required interface
var _ bootstrap.SystemContainer = &systemContainerImpl{}

// systemContainerImpl implements bootstrap.SystemContainer
type systemContainerImpl struct {
	name   string
	limits api.ResourceList
}

// Name returns the absolute name of the container
func (s *systemContainerImpl) Name() string {
	return s.name
}

// Limits is the set of resources allocated to the system container.
func (s *systemContainerImpl) Limits() api.ResourceList {
	return s.limits
}

// ensure we implement the required interface
var _ bootstrap.BootstrapManager = &bootstrapManagerImpl{}

// bootstrapManagerImpl implements bootstrap.BootstrapManager for systemd environments
type bootstrapManagerImpl struct {
	cadvisorInterface cadvisor.Interface
	bootstrap.NodeConfig
	mountUtil        mount.Interface
	systemContainers []bootstrap.SystemContainer
}

// NewBootstrapManager creates a manager for systemd systems
func NewBootstrapManager(mountUtil mount.Interface, cadvisorInterface cadvisor.Interface) (bootstrap.BootstrapManager, error) {
	return &bootstrapManagerImpl{
		cadvisorInterface: cadvisorInterface,
		NodeConfig:        bootstrap.NodeConfig{},
		mountUtil:         mountUtil,
	}, nil
}

// Start initializes the node for systemd systems
func (bm *bootstrapManagerImpl) Start(nodeConfig bootstrap.NodeConfig) error {
	bm.NodeConfig = nodeConfig
	systemContainers := []bootstrap.SystemContainer{}

	glog.Info("Bootstrapping container manager for systemd.")
	if len(bm.KubeletContainerName) != 0 {
		glog.Warningf("Container manager ignores user provided kubelet container name on systemd: %v", bm.KubeletContainerName)
	}

	// find the cgroup that manages the kubelet
	ownCgroup, err := findOwnCgroup()
	if err != nil {
		return err
	}
	container := &systemContainerImpl{name: ownCgroup, limits: api.ResourceList{}}
	systemContainers = append(systemContainers, container)
	glog.Infof("Container manager has detected the kubelet is running in container: %v", ownCgroup)

	// on systemd, the default system container is system.slice
	systemContainer := "/system.slice"
	if len(bm.NodeConfig.SystemContainerName) != 0 {
		systemContainer = bm.NodeConfig.SystemContainerName
	}
	container = &systemContainerImpl{name: systemContainer, limits: api.ResourceList{}}
	systemContainers = append(systemContainers, container)
	glog.Infof("Container manager using system-container: %v", systemContainer)

	bm.systemContainers = systemContainers
	return nil
}

// SystemContainers is the list of non-user containers managed during bootstrapping.
func (bm *bootstrapManagerImpl) SystemContainers() []bootstrap.SystemContainer {
	return bm.systemContainers
}

// findOwnCgroup looks for the cgroup of the current process
func findOwnCgroup() (string, error) {
	cgs, err := cgroups.ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	selfCgroup, found := cgs["name=systemd"]
	if !found {
		return "", fmt.Errorf("Unable to find name=systemd in /proc/self/cgroup")
	}
	return selfCgroup, nil
}
