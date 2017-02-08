/*
Copyright 2015 The Kubernetes Authors.

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
	"strconv"
	"strings"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
)

// Manages the containers running on a machine.
type ContainerManager interface {
	// Runs the container manager's housekeeping.
	// - Ensures that the Docker daemon is in a container.
	// - Creates the system container where all non-containerized processes run.
	Start(*v1.Node, func() []*v1.Pod) error

	// Returns resources allocated to system cgroups in the machine.
	// These cgroups include the system and Kubernetes services.
	SystemCgroupsLimit() v1.ResourceList

	// Returns a NodeConfig that is being used by the container manager.
	GetNodeConfig() NodeConfig

	// Returns internal Status.
	Status() Status

	// NewPodContainerManager is a factory method which returns a podContainerManager object
	// Returns a noop implementation if qos cgroup hierarchy is not enabled
	NewPodContainerManager() PodContainerManager

	// GetMountedSubsystems returns the mounted cgroup subsytems on the node
	GetMountedSubsystems() *CgroupSubsystems

	// GetQOSContainersInfo returns the names of top level QoS containers
	GetQOSContainersInfo() QOSContainersInfo
}

type QOSReserveLimit struct {
	resource       string
	reservePercent int64
}

type NodeConfig struct {
	RuntimeCgroupsName    string
	SystemCgroupsName     string
	KubeletCgroupsName    string
	ContainerRuntime      string
	CgroupsPerQOS         bool
	CgroupRoot            string
	CgroupDriver          string
	ProtectKernelDefaults bool
	EnableCRI             bool
	QOSReserveLimits      []QOSReserveLimit
}

type Status struct {
	// Any soft requirements that were unsatisfied.
	SoftRequirements error
}

func parseQOSReserveLimit(statement string) (*QOSReserveLimit, error) {
	supportedResources := sets.NewString("memory")
	parts := strings.Split(statement, "=")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid statement in QoS reserve limit option")
	}
	resource := parts[0]
	if !supportedResources.Has(resource) {
		return nil, fmt.Errorf("unsupported resource '%s' in QoS reserve limit option", resource)
	}
	reservePercentStr := parts[1]
	if !strings.HasSuffix(parts[1], "%") {
		return nil, fmt.Errorf("invalid percentage '%s' in QoS reserve limit option", reservePercentStr)
	}
	reservePercent, err := strconv.ParseInt(strings.TrimRight(reservePercentStr, "%"), 10, 0)
	if err != nil {
		return nil, fmt.Errorf("invalid number '%s' in QoS reserve limit option", reservePercentStr)
	}
	if reservePercent < 0 || reservePercent > 100 {
		return nil, fmt.Errorf("percent must be between 0 and 100 in QoS reserve limit option")
	}
	glog.V(2).Infof("QoS reserve limits for %s configured at %s", resource, reservePercentStr)
	return &QOSReserveLimit{
		resource:       resource,
		reservePercent: reservePercent,
	}, nil
}

// ParseQOSReserveLimits parses the --qos-reserve-limits kubelet option
func ParseQOSReserveLimits(expr string) ([]QOSReserveLimit, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := []QOSReserveLimit{}
	statements := strings.Split(expr, ",")
	resourcesFound := sets.NewString()
	for _, statement := range statements {
		reserveLimit, err := parseQOSReserveLimit(statement)
		if err != nil {
			return nil, err
		}
		if resourcesFound.Has(reserveLimit.resource) {
			return nil, fmt.Errorf("found duplicate resource reserve limit for '%v' in QoS reserve limit option", reserveLimit.resource)
		}
		resourcesFound.Insert(reserveLimit.resource)
		results = append(results, *reserveLimit)
	}
	return results, nil
}
