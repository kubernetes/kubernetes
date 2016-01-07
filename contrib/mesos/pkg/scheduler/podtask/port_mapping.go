/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package podtask

import (
	"fmt"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

const (
	// maps a Container.HostPort to the same exact offered host port, ignores .HostPort = 0
	HostPortMappingFixed = "fixed"
	// same as HostPortMappingFixed, except that .HostPort of 0 are mapped to any port offered
	HostPortMappingWildcard = "wildcard"
)

// Objects implementing the HostPortMapper interface generate port mappings
// from k8s container ports to ports offered by mesos
type HostPortMapper interface {
	// Map maps the given pod task and the given mesos offer
	// and returns a slice of port mappings
	// or an error if the mapping failed
	Map(t *T, offer *mesos.Offer) ([]HostPortMapping, error)
}

// HostPortMapperFunc is a function adapter to the HostPortMapper interface
type HostPortMapperFunc func(*T, *mesos.Offer) ([]HostPortMapping, error)

// Map calls f(t, offer)
func (f HostPortMapperFunc) Map(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
	return f(t, offer)
}

// A HostPortMapping represents the mapping between k8s container ports
// ports offered by mesos. It references the k8s' container and port
// and specifies the offered mesos port and the offered port's role
type HostPortMapping struct {
	ContainerIdx int    // index of the container in the pod spec
	PortIdx      int    // index of the port in a container's port spec
	OfferPort    uint64 // the port offered by mesos
	Role         string // the role asssociated with the offered port
}

type PortAllocationError struct {
	PodId string
	Ports []uint64
}

func (err *PortAllocationError) Error() string {
	return fmt.Sprintf("Could not schedule pod %s: %d port(s) could not be allocated", err.PodId, len(err.Ports))
}

type DuplicateHostPortError struct {
	m1, m2 HostPortMapping
}

func (err *DuplicateHostPortError) Error() string {
	return fmt.Sprintf(
		"Host port %d is specified for container %d, pod %d and container %d, pod %d",
		err.m1.OfferPort, err.m1.ContainerIdx, err.m1.PortIdx, err.m2.ContainerIdx, err.m2.PortIdx)
}

// WildcardMapper maps k8s wildcard ports (hostPort == 0) to any available offer port
func WildcardMapper(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
	mapping, err := FixedMapper(t, offer)
	if err != nil {
		return nil, err
	}

	taken := make(map[uint64]struct{})
	for _, entry := range mapping {
		taken[entry.OfferPort] = struct{}{}
	}

	wildports := []HostPortMapping{}
	for i, container := range t.Pod.Spec.Containers {
		for pi, port := range container.Ports {
			if port.HostPort == 0 {
				wildports = append(wildports, HostPortMapping{
					ContainerIdx: i,
					PortIdx:      pi,
				})
			}
		}
	}

	remaining := len(wildports)
	foreachPortsRange(offer.GetResources(), t.Roles(), func(bp, ep uint64, role string) {
		log.V(3).Infof("Searching for wildcard port in range {%d:%d}", bp, ep)
		for i := range wildports {
			if wildports[i].OfferPort != 0 {
				continue
			}
			for port := bp; port <= ep && remaining > 0; port++ {
				if _, inuse := taken[port]; inuse {
					continue
				}
				wildports[i].OfferPort = port
				wildports[i].Role = starredRole(role)
				mapping = append(mapping, wildports[i])
				remaining--
				taken[port] = struct{}{}
				break
			}
		}
	})

	if remaining > 0 {
		err := &PortAllocationError{
			PodId: t.Pod.Name,
		}
		// it doesn't make sense to include a port list here because they were all zero (wildcards)
		return nil, err
	}

	return mapping, nil
}

// FixedMapper maps k8s host ports to offered ports ignoring hostPorts == 0 (remaining pod-private)
func FixedMapper(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
	requiredPorts := make(map[uint64]HostPortMapping)
	mapping := []HostPortMapping{}
	for i, container := range t.Pod.Spec.Containers {
		// strip all port==0 from this array; k8s already knows what to do with zero-
		// ports (it does not create 'port bindings' on the minion-host); we need to
		// remove the wildcards from this array since they don't consume host resources
		for pi, port := range container.Ports {
			if port.HostPort == 0 {
				continue // ignore
			}
			m := HostPortMapping{
				ContainerIdx: i,
				PortIdx:      pi,
				OfferPort:    uint64(port.HostPort),
			}
			if entry, inuse := requiredPorts[uint64(port.HostPort)]; inuse {
				return nil, &DuplicateHostPortError{entry, m}
			}
			requiredPorts[uint64(port.HostPort)] = m
		}
	}

	foreachPortsRange(offer.GetResources(), t.Roles(), func(bp, ep uint64, role string) {
		for port := range requiredPorts {
			log.V(3).Infof("evaluating port range {%d:%d} %d", bp, ep, port)
			if (bp <= port) && (port <= ep) {
				m := requiredPorts[port]
				m.Role = starredRole(role)
				mapping = append(mapping, m)
				delete(requiredPorts, port)
			}
		}
	})

	unsatisfiedPorts := len(requiredPorts)
	if unsatisfiedPorts > 0 {
		err := &PortAllocationError{
			PodId: t.Pod.Name,
		}
		for p := range requiredPorts {
			err.Ports = append(err.Ports, p)
		}
		return nil, err
	}

	return mapping, nil
}

// NewHostPortMapper returns a new mapper based
// based on the port mapping key value
func NewHostPortMapper(pod *api.Pod) HostPortMapper {
	filter := map[string]string{
		meta.PortMappingKey: HostPortMappingFixed,
	}
	selector := labels.Set(filter).AsSelector()
	if selector.Matches(labels.Set(pod.Labels)) {
		return HostPortMapperFunc(FixedMapper)
	}
	return HostPortMapperFunc(WildcardMapper)
}
