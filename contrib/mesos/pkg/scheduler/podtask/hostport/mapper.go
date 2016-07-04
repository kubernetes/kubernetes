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

package hostport

import (
	"fmt"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
)

// Objects implementing the Mapper interface generate port mappings
// from k8s container ports to ports offered by mesos
type Mapper interface {
	// Map maps the given pod and the given mesos offer and returns a
	// slice of port mappings or an error if the mapping failed
	Map(pod *api.Pod, roles []string, offer *mesos.Offer) ([]Mapping, error)
}

// MapperFunc is a function adapter to the Mapper interface
type MapperFunc func(*api.Pod, []string, *mesos.Offer) ([]Mapping, error)

// Map calls f(t, offer)
func (f MapperFunc) Map(pod *api.Pod, roles []string, offer *mesos.Offer) ([]Mapping, error) {
	return f(pod, roles, offer)
}

// A Mapping represents the mapping between k8s container ports
// ports offered by mesos. It references the k8s' container and port
// and specifies the offered mesos port and the offered port's role
type Mapping struct {
	ContainerIdx int    // index of the container in the pod spec
	PortIdx      int    // index of the port in a container's port spec
	OfferPort    uint64 // the port offered by mesos
	Role         string // the role asssociated with the offered port
}

type PortAllocationError struct {
	PodID string
	Ports []uint64
}

func (err *PortAllocationError) Error() string {
	return fmt.Sprintf("Could not schedule pod %s: %d port(s) could not be allocated", err.PodID, len(err.Ports))
}

type DuplicateError struct {
	m1, m2 Mapping
}

func (err *DuplicateError) Error() string {
	return fmt.Sprintf(
		"Host port %d is specified for container %d, pod %d and container %d, pod %d",
		err.m1.OfferPort, err.m1.ContainerIdx, err.m1.PortIdx, err.m2.ContainerIdx, err.m2.PortIdx)
}

// WildcardMapper maps k8s wildcard ports (hostPort == 0) to any available offer port
func WildcardMapper(pod *api.Pod, roles []string, offer *mesos.Offer) ([]Mapping, error) {
	mapping, err := FixedMapper(pod, roles, offer)
	if err != nil {
		return nil, err
	}

	taken := make(map[uint64]struct{})
	for _, entry := range mapping {
		taken[entry.OfferPort] = struct{}{}
	}

	wildports := []Mapping{}
	for i, container := range pod.Spec.Containers {
		for pi, port := range container.Ports {
			if port.HostPort == 0 {
				wildports = append(wildports, Mapping{
					ContainerIdx: i,
					PortIdx:      pi,
				})
			}
		}
	}

	remaining := len(wildports)
	resources.ForeachPortsRange(offer.GetResources(), roles, func(bp, ep uint64, role string) {
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
				wildports[i].Role = resources.CanonicalRole(role)
				mapping = append(mapping, wildports[i])
				remaining--
				taken[port] = struct{}{}
				break
			}
		}
	})

	if remaining > 0 {
		err := &PortAllocationError{
			PodID: pod.Namespace + "/" + pod.Name,
		}
		// it doesn't make sense to include a port list here because they were all zero (wildcards)
		return nil, err
	}

	return mapping, nil
}

// FixedMapper maps k8s host ports to offered ports ignoring hostPorts == 0 (remaining pod-private)
func FixedMapper(pod *api.Pod, roles []string, offer *mesos.Offer) ([]Mapping, error) {
	requiredPorts := make(map[uint64]Mapping)
	mapping := []Mapping{}
	for i, container := range pod.Spec.Containers {
		// strip all port==0 from this array; k8s already knows what to do with zero-
		// ports (it does not create 'port bindings' on the minion-host); we need to
		// remove the wildcards from this array since they don't consume host resources
		for pi, port := range container.Ports {
			if port.HostPort == 0 {
				continue // ignore
			}
			m := Mapping{
				ContainerIdx: i,
				PortIdx:      pi,
				OfferPort:    uint64(port.HostPort),
			}
			if entry, inuse := requiredPorts[uint64(port.HostPort)]; inuse {
				return nil, &DuplicateError{entry, m}
			}
			requiredPorts[uint64(port.HostPort)] = m
		}
	}

	resources.ForeachPortsRange(offer.GetResources(), roles, func(bp, ep uint64, role string) {
		for port := range requiredPorts {
			log.V(3).Infof("evaluating port range {%d:%d} %d", bp, ep, port)
			if (bp <= port) && (port <= ep) {
				m := requiredPorts[port]
				m.Role = resources.CanonicalRole(role)
				mapping = append(mapping, m)
				delete(requiredPorts, port)
			}
		}
	})

	unsatisfiedPorts := len(requiredPorts)
	if unsatisfiedPorts > 0 {
		err := &PortAllocationError{
			PodID: pod.Namespace + "/" + pod.Name,
		}
		for p := range requiredPorts {
			err.Ports = append(err.Ports, p)
		}
		return nil, err
	}

	return mapping, nil
}

type Strategy string

const (
	// maps a Container.HostPort to the same exact offered host port, ignores .HostPort = 0
	StrategyFixed = Strategy("fixed")
	// same as MappingFixed, except that .HostPort of 0 are mapped to any port offered
	StrategyWildcard = Strategy("wildcard")
)

var validStrategies = map[Strategy]MapperFunc{
	StrategyFixed:    MapperFunc(FixedMapper),
	StrategyWildcard: MapperFunc(WildcardMapper),
}

// NewMapper returns a new mapper based on the port mapping key value
func (defaultStrategy Strategy) NewMapper(pod *api.Pod) Mapper {
	strategy, ok := pod.Labels[meta.PortMappingKey]
	if ok {
		f, ok := validStrategies[Strategy(strategy)]
		if ok {
			return f
		}
		log.Warningf("invalid port mapping strategy %q, reverting to default %q", strategy, defaultStrategy)
	}

	f, ok := validStrategies[defaultStrategy]
	if ok {
		return f
	}

	panic("scheduler is misconfigured, unrecognized default strategy \"" + defaultStrategy + "\"")
}
