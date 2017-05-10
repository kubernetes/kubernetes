/*
Copyright 2017 The Kubernetes Authors.

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

package coreaffinity

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity/cputopology"
	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity/discovery"
	"k8s.io/kubernetes/cluster/addons/iso-client/opaque"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type coreAffinityIsolator struct {
	name             string
	cpuAssignmentMap map[string][]int
	CPUTopology      *cputopology.CPUTopology
}

// implementation of Init() method in  "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle".Isolator
func (c *coreAffinityIsolator) Init() error {
	return opaque.AdvertiseOpaqueResource(c.name, c.CPUTopology.GetTotalCPUs())
}

// Constructor for Isolator
func New(name string) (*coreAffinityIsolator, error) {
	topology, err := discovery.DiscoverTopology()
	return &coreAffinityIsolator{
		name:             name,
		CPUTopology:      topology,
		cpuAssignmentMap: make(map[string][]int),
	}, err
}

func (c *coreAffinityIsolator) gatherContainerRequest(container v1.Container) int64 {
	resource, ok := container.Resources.Requests[helper.OpaqueIntResourceName(c.name)]
	if !ok {
		return 0
	}
	return resource.Value()
}

func (c *coreAffinityIsolator) countCoresFromOIR(pod *v1.Pod) int64 {
	var coresAccu int64
	for _, container := range pod.Spec.Containers {
		coresAccu = coresAccu + c.gatherContainerRequest(container)
	}
	return coresAccu
}

func (c *coreAffinityIsolator) reserveCPUs(cores int64) ([]int, error) {
	cpus := c.CPUTopology.GetAvailableCPUs()
	if len(cpus) < int(cores) {
		return nil, fmt.Errorf("cannot reserved requested number of cores")
	}
	var reservedCores []int

	for idx := 0; idx < int(cores); idx++ {
		if err := c.CPUTopology.Reserve(cpus[idx]); err != nil {
			return reservedCores, err
		}
		reservedCores = append(reservedCores, cpus[idx])
	}
	return reservedCores, nil

}

func (c *coreAffinityIsolator) reclaimCPUs(cores []int) {
	for _, core := range cores {
		c.CPUTopology.Reclaim(core)
	}
}

func asCPUList(cores []int) string {
	var coresStr []string
	for _, core := range cores {
		coresStr = append(coresStr, strconv.Itoa(core))
	}
	return strings.Join(coresStr, ",")
}

// implementation of preStart method in  "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle".Isolator
func (c *coreAffinityIsolator) PreStartPod(podName string, containerName string, pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error) {
	oirCores := c.countCoresFromOIR(pod)
	glog.Infof("Pod %s requested %d cores", pod.Name, oirCores)

	if oirCores == 0 {
		glog.Infof("Pod %q isn't managed by this isolator", pod.Name)
		return []*lifecycle.IsolationControl{}, nil
	}

	reservedCores, err := c.reserveCPUs(oirCores)
	if err != nil {
		c.reclaimCPUs(reservedCores)
		return []*lifecycle.IsolationControl{}, err
	}

	cgroupResource := []*lifecycle.IsolationControl{
		{
			Value: asCPUList(reservedCores),
			Kind:  lifecycle.IsolationControl_CGROUP_CPUSET_CPUS,
		},
	}
	c.cpuAssignmentMap[resource.Path] = reservedCores
	return cgroupResource, nil
}

// implementation of postStop method in "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle".Isolator
func (c *coreAffinityIsolator) PostStopPod(podName string, containerName string, cgroupInfo *lifecycle.CgroupInfo) error {
	cpus := c.cpuAssignmentMap[cgroupInfo.Path]
	c.reclaimCPUs(cpus)
	delete(c.cpuAssignmentMap, cgroupInfo.Path)
	return nil
}

func (c *coreAffinityIsolator) PreStartContainer(podName, containerName string) ([]*lifecycle.IsolationControl, error) {
	glog.Infof("coreAffinityIsolator[PreStartContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return []*lifecycle.IsolationControl{}, nil
}

func (c *coreAffinityIsolator) PostStopContainer(podName, containerName string) error {
	glog.Infof("coreAffinityIsolator[PostStopContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return nil
}

func (c *coreAffinityIsolator) ShutDown() {
	opaque.RemoveOpaqueResource(c.name)
}

// implementation of Name method in "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle".Isolator
func (c *coreAffinityIsolator) Name() string {
	return c.name
}
