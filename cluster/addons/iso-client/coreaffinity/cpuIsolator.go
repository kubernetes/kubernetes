package coreaffinity

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

type cpuIsolator struct {
	Name             string
	CPUTopology      *cputopology.CPUTopology
	cpuAssignmentMap map[string][]int
}

// Constructor for Isolator
func NewIsolator(name string) (*cpuIsolator, error) {
	topology, err := discovery.DiscoverTopology()
	return &cpuIsolator{
		Name:             name,
		CPUTopology:      topology,
		cpuAssignmentMap: make(map[string][]int),
	}, err
}

func (i *cpuIsolator) gatherContainerRequest(container v1.Container) int64 {
	resource, ok := container.Resources.Requests[v1.OpaqueIntResourceName(i.Name)]
	if !ok {
		return 0
	}
	return resource.Value()
}

func (i *cpuIsolator) countCoresFromOIR(pod *v1.Pod) int64 {
	var coresAccu int64
	for _, container := range pod.Spec.Containers {
		coresAccu = coresAccu + i.gatherContainerRequest(container)
	}
	return coresAccu
}

func (i *cpuIsolator) reserveCPUs(cores int64) ([]int, error) {
	cpus := i.CPUTopology.GetAvailableCPUs()
	if len(cpus) < int(cores) {
		return nil, fmt.Errorf("cannot reserved requested number of cores")
	}
	var reservedCores []int

	for idx := 0; idx < int(cores); idx++ {
		if err := i.CPUTopology.Reserve(cpus[idx]); err != nil {
			return reservedCores, err
		}
		reservedCores = append(reservedCores, cpus[idx])
	}
	return reservedCores, nil

}

func (i *cpuIsolator) reclaimCPUs(cores []int) {
	for _, core := range cores {
		i.CPUTopology.Reclaim(core)
	}
}

func asCPUList(cores []int) string {
	var coresStr []string
	for _, core := range cores {
		coresStr = append(coresStr, strconv.Itoa(core))
	}
	return strings.Join(coresStr, ",")
}

func (i *cpuIsolator) PreStart(pod *v1.Pod, resource *lifecycle.CgroupInfo) ([]*lifecycle.CgroupResource, error) {
	oirCores := i.countCoresFromOIR(pod)
	glog.Infof("Pod %s requested %d cores", pod.Name, oirCores)

	if oirCores == 0 {
		glog.Infof("Pod %q isn't managed by this isolator", pod.Name)
		return []*lifecycle.CgroupResource{}, nil
	}

	reservedCores, err := i.reserveCPUs(oirCores)
	if err != nil {
		i.reclaimCPUs(reservedCores)
		return []*lifecycle.CgroupResource{}, err
	}

	cgroupResource := []*lifecycle.CgroupResource{
		{
			Value:           asCPUList(reservedCores),
			CgroupSubsystem: lifecycle.CgroupResource_CPUSET_CPUS,
		},
	}
	i.cpuAssignmentMap[resource.Path] = reservedCores
	return cgroupResource, nil
}

func (i *cpuIsolator) PostStop(cgroupInfo *lifecycle.CgroupInfo) error {
	cpus := i.cpuAssignmentMap[cgroupInfo.Path]
	i.reclaimCPUs(cpus)
	delete(i.cpuAssignmentMap, cgroupInfo.Path)
	return nil
}
