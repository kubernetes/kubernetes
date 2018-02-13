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

package proxy

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// ProxyProvider is the interface provided by proxier implementations.
type ProxyProvider interface {
	// Sync immediately synchronizes the ProxyProvider's current state to proxy rules.
	Sync()
	// SyncLoop runs periodic work.
	// This is expected to run as a goroutine or as the main loop of the app.
	// It does not return.
	SyncLoop()
}

// ServicePortName carries a namespace + name + portname.  This is the unique
// identifier for a load-balanced service.
type ServicePortName struct {
	types.NamespacedName
	Port string
}

func (spn ServicePortName) String() string {
	return fmt.Sprintf("%s:%s", spn.NamespacedName.String(), spn.Port)
}

type ServicePort interface {
	// returns service string
	String() string
	// returns service cluster IP
	ClusterIP() string
	// returns service protocol
	Protocol() api.Protocol
	// returns service health check node port if present
	HealthCheckNodePort() int
}

type Endpoint interface {
	// returns endpoint string
	String() string
	// returns true if the endpoint is running in same host as kube-proxy, otherwise returns false
	IsLocal() bool
	// returns IP part of endpoints
	IP() string
	// checks if two endpoints are equal
	Equal(Endpoint) bool
}

// ServiceEndpoint is used to identify a service and one of its endpoint pair.
type ServiceEndpoint struct {
	Endpoint        string
	ServicePortName ServicePortName
}
