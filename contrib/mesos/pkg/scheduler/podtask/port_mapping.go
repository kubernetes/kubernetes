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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

type HostPortMappingType string

const (
	// maps a Container.HostPort to the same exact offered host port, ignores .HostPort = 0
	HostPortMappingFixed HostPortMappingType = "fixed"
	// same as HostPortMappingFixed, except that .HostPort of 0 are mapped to any port offered
	HostPortMappingWildcard = "wildcard"
)

type HostPortMapper interface {
	// abstracts the way that host ports are mapped to pod container ports
	Generate(t *T, offer *mesos.Offer) ([]HostPortMapping, error)
}

type HostPortMapping struct {
	ContainerIdx int // index of the container in the pod spec
	PortIdx      int // index of the port in a container's port spec
	OfferPort    uint64
}

func (self HostPortMappingType) Generate(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
	switch self {
	case HostPortMappingWildcard:
		return wildcardHostPortMapping(t, offer)
	case HostPortMappingFixed:
	default:
		log.Warningf("illegal host-port mapping spec %q, defaulting to %q", self, HostPortMappingFixed)
	}
	return defaultHostPortMapping(t, offer)
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

// wildcard k8s host port mapping implementation: hostPort == 0 gets mapped to any available offer port
func wildcardHostPortMapping(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
	mapping, err := defaultHostPortMapping(t, offer)
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
	foreachRange(offer, "ports", func(bp, ep uint64) {
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

// default k8s host port mapping implementation: hostPort == 0 means containerPort remains pod-private, and so
// no offer ports will be mapped to such Container ports.
func defaultHostPortMapping(t *T, offer *mesos.Offer) ([]HostPortMapping, error) {
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
	foreachRange(offer, "ports", func(bp, ep uint64) {
		for port := range requiredPorts {
			log.V(3).Infof("evaluating port range {%d:%d} %d", bp, ep, port)
			if (bp <= port) && (port <= ep) {
				mapping = append(mapping, requiredPorts[port])
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

const PortMappingLabelKey = "k8s.mesosphere.io/portMapping"

func MappingTypeForPod(pod *api.Pod) HostPortMappingType {
	filter := map[string]string{
		PortMappingLabelKey: string(HostPortMappingFixed),
	}
	selector := labels.Set(filter).AsSelector()
	if selector.Matches(labels.Set(pod.Labels)) {
		return HostPortMappingFixed
	}
	return HostPortMappingWildcard
}
