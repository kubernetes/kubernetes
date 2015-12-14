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

package concerto_cloud

import (
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// ProviderName returns the cloud provider ID.
func (f *ConcertoCloud) ProviderName() string {
	return ProviderName
}

// Instances returns an implementation of Instances for Concerto.
func (concerto *ConcertoCloud) Instances() (cloudprovider.Instances, bool) {
	return concerto, true
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Concerto.
func (concerto *ConcertoCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return concerto, true
}

// Clusters not supported.
func (concerto *ConcertoCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// Routes not supported.
func (concerto *ConcertoCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for Concerto.
func (concerto *ConcertoCloud) Zones() (cloudprovider.Zones, bool) {
	return concerto, true
}

// ScrubDNS filters DNS settings for pods.
func (concerto *ConcertoCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}
