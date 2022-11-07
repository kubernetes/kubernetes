/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
)

// PortsByPodUID is a map that maps pod UID to container ports.
type PortsByPodUID map[types.UID][]int

// FullPortsByPodUID is a map that maps pod UID to container ports.
type FullPortsByPodUID map[types.UID][]v1.ContainerPort

// GetContainerPortsByPodUID returns a PortsByPodUID map on the given endpoints.
func GetContainerPortsByPodUID(eps []discoveryv1.EndpointSlice) PortsByPodUID {
	m := PortsByPodUID{}

	for _, es := range eps {
		for _, port := range es.Ports {
			if port.Port == nil {
				continue
			}
			for _, ep := range es.Endpoints {
				containerPort := *port.Port
				if _, ok := m[ep.TargetRef.UID]; !ok {
					m[ep.TargetRef.UID] = make([]int, 0)
				}
				m[ep.TargetRef.UID] = append(m[ep.TargetRef.UID], int(containerPort))
			}
		}
	}
	return m
}

// GetFullContainerPortsByPodUID returns a PortsByPodUID map on the given endpoints.
func GetFullContainerPortsByPodUID(eps []discoveryv1.EndpointSlice) FullPortsByPodUID {
	m := FullPortsByPodUID{}

	for _, es := range eps {
		for _, port := range es.Ports {
			if port.Port == nil {
				continue
			}
			containerPort := v1.ContainerPort{
				Name:          *port.Name,
				ContainerPort: *port.Port,
				Protocol:      *port.Protocol,
			}
			for _, ep := range es.Endpoints {
				if _, ok := m[ep.TargetRef.UID]; !ok {
					m[ep.TargetRef.UID] = make([]v1.ContainerPort, 0)
				}
				m[ep.TargetRef.UID] = append(m[ep.TargetRef.UID], containerPort)
			}
		}
	}
	return m
}
