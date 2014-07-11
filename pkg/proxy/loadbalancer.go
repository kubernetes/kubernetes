/*
Copyright 2014 Google Inc. All rights reserved.

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

// Loadbalancer interface. Implementations use loadbalancer_<strategy> naming.

package proxy

import (
	"net"
)

// LoadBalancer represents a load balancer that decides where to route
// the incoming services for a particular service to.
type LoadBalancer interface {
	// LoadBalance takes an incoming request and figures out where to route it to.
	// Determination is based on destination service (for example, 'mysql') as
	// well as the source making the connection.
	LoadBalance(service string, srcAddr net.Addr) (string, error)
}
