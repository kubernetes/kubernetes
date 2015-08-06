/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package proxy

import (
	"fmt"
	"net"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
)

// LoadBalancer is an interface for distributing incoming requests to service endpoints.
type LoadBalancer interface {
	// NextEndpoint returns the endpoint to handle a request for the given
	// service-port and source address.
	NextEndpoint(service ServicePortName, srcAddr net.Addr) (string, error)
	NewService(service ServicePortName, sessionAffinityType api.ServiceAffinity, stickyMaxAgeMinutes int) error
	CleanupStaleStickySessions(service ServicePortName)
}

// ServicePortName carries a namespace + name + portname.  This is the unique
// identfier for a load-balanced service.
type ServicePortName struct {
	types.NamespacedName
	Port string
}

func (spn ServicePortName) String() string {
	return fmt.Sprintf("%s:%s", spn.NamespacedName.String(), spn.Port)
}
