// +build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/qos"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

const (
	// The percent of the machine memory capacity.
	dockerMemoryLimitThresholdPercent = kubecm.DockerMemoryLimitThresholdPercent

	// The minimum memory limit allocated to docker container.
	minDockerMemoryLimit = kubecm.MinDockerMemoryLimit

	// The Docker OOM score adjustment.
	dockerOOMScoreAdj = qos.DockerOOMScoreAdj
)

var (
	memoryCapacityRegexp = regexp.MustCompile(`MemTotal:\s*([0-9]+) kB`)
)

// NewContainerManager creates a new instance of ContainerManager
func NewContainerManager(cgroupsName string, client libdocker.Interface) ContainerManager {
	return &containerManager{
		cgroupsName: cgroupsName,
		client:      client,
	}
}

type containerManager struct {
	// Docker client.
	client libdocker.Interface
	// Name of the cgroups.
	cgroupsName string
	// Manager for the cgroups.
	cgroupsManager *fs.Manager
}

func (m *containerManager) Start() error {
	// TODO: check if the required cgroups are mounted.
	if len(m.cgroupsName) != 0 {
		manager, err := createCgroupManager(m.cgroupsName)
		if err != nil {
			return err
		}
		m.cgroupsManager = manager
	}
	go wait.Until(m.doWork, 5*time.Minute, wait.NeverStop)
	return nil
}

func (m *containerManager) doWork() {
	v, err := m.client.Version()
	if err != nil {
		klog.Errorf("Unable to get docker version: %v", err)
		return
	}
	version, err := utilversion.ParseGeneric(v.APIVersion)
	if err != nil {
		klog.Errorf("Unable to parse docker version %q: %v", v.APIVersion, err)
		return
	}
	// EnsureDockerInContainer does two things.
	//   1. Ensure processes run in the cgroups if m.cgroupsManager is not nil.
	//   2. Ensure processes have the OOM score applied.
	if err := kubecm.EnsureDockerInContainer(version, dockerOOMScoreAdj, m.cgroupsManager); err != nil {
		klog.Errorf("Unable to ensure the docker processes run in the desired containers: %v", err)
	}
}

func createCgroupManager(name string) (*fs.Manager, error) {
	var memoryLimit uint64

	memoryCapacity, err := getMemoryCapacity()
	if err != nil {
		klog.Errorf("Failed to get the memory capacity on machine: %v", err)
	} else {
		memoryLimit = memoryCapacity * dockerMemoryLimitThresholdPercent / 100
	}

	if err != nil || memoryLimit < minDockerMemoryLimit {
		memoryLimit = minDockerMemoryLimit
	}
	klog.V(2).Infof("Configure resource-only container %q with memory limit: %d", name, memoryLimit)

	allowAllDevices := true
	cm := &fs.Manager{
		Cgroups: &configs.Cgroup{
			Parent: "/",
			Name:   name,
			Resources: &configs.Resources{
				Memory:          int64(memoryLimit),
				MemorySwap:      -1,
				AllowAllDevices: &allowAllDevices,
			},
		},
	}
	return cm, nil
}

// getMemoryCapacity returns the memory capacity on the machine in bytes.
func getMemoryCapacity() (uint64, error) {
	out, err := ioutil.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	return parseCapacity(out, memoryCapacityRegexp)
}

// parseCapacity matches a Regexp in a []byte, returning the resulting value in bytes.
// Assumes that the value matched by the Regexp is in KB.
func parseCapacity(b []byte, r *regexp.Regexp) (uint64, error) {
	matches := r.FindSubmatch(b)
	if len(matches) != 2 {
		return 0, fmt.Errorf("failed to match regexp in output: %q", string(b))
	}
	m, err := strconv.ParseUint(string(matches[1]), 10, 64)
	if err != nil {
		return 0, err
	}

	// Convert to bytes.
	return m * 1024, err
}
