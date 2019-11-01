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
	discovery "k8s.io/api/discovery/v1alpha1"
)

// EndpointsByPort groups lists of Endpoints by EndpointPort.
type EndpointsByPort map[discovery.EndpointPort][]discovery.Endpoint

// GetEndpointsByPort accepts a list of EndpointSlices and returns
// EndpointsByPort, lists of Endpoints grouped by EndpointPort.
func GetEndpointsByPort(endpointSlices []*discovery.EndpointSlice) EndpointsByPort {
	endpointsByPort := EndpointsByPort{}
	for _, endpointSlice := range endpointSlices {
		for _, port := range endpointSlice.Ports {
			endpointsByPort[port] = append(endpointsByPort[port], endpointSlice.Endpoints...)
		}
	}
	return endpointsByPort
}

// GetEndpointsForPort accepts a port name along with a list of EndpointSlices
// and returns EndpointsByPort. Since a given port name may result in multiple
// port numbers, Endpoints are returned as a map grouped by EndpointPort.
func GetEndpointsForPort(portName string, endpointSlices []*discovery.EndpointSlice) EndpointsByPort {
	endpointsByPort := EndpointsByPort{}
	for _, endpointSlice := range endpointSlices {
		for _, port := range endpointSlice.Ports {
			if port.Name != nil && *port.Name == portName {
				endpointsByPort[port] = append(endpointsByPort[port], endpointSlice.Endpoints...)
			}
		}
	}
	return endpointsByPort
}
