package coreaffinityisolator

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity/cputopology"
	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity/discovery"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type coreAffinityIsolator struct {
	Name             string
	cpuAssignmentMap map[string][]int
	CPUTopology      *cputopology.CPUTopology
}

// Constructor for Isolator
func New(name string) (*coreAffinityIsolator, error) {
	topology, err := discovery.DiscoverTopology()
	return &coreAffinityIsolator{
		Name:             name,
		CPUTopology:      topology,
		cpuAssignmentMap: make(map[string][]int),
	}, err
}

func (c *coreAffinityIsolator) gatherContainerRequest(container v1.Container) int64 {
	resource, ok := container.Resources.Requests[v1.OpaqueIntResourceName(c.Name)]
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

// implementation of preStart method in  isolator Interface
func (c *coreAffinityIsolator) PreStart(pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error) {
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

// implementation of postStop method in isolator Interface
func (c *coreAffinityIsolator) PostStop(cgroupInfo *lifecycle.CgroupInfo) error {
	cpus := c.cpuAssignmentMap[cgroupInfo.Path]
	c.reclaimCPUs(cpus)
	delete(c.cpuAssignmentMap, cgroupInfo.Path)
	return nil
}
