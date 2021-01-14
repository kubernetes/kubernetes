// +build !dockerless

/*
Copyright 2014 The Kubernetes Authors.

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

package testing

// helper for testing plugins
// a fake host is created here that can be used by plugins for testing

import (
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport"
)

// FakeNetworkHost is the struct for testing
type FakeNetworkHost struct {
	fakeNamespaceGetter
	FakePortMappingGetter
	kubeClient clientset.Interface
	Legacy     bool
}

// NewFakeHost returns a struct that implements the FakeNetworkHost interface
func NewFakeHost(kubeClient clientset.Interface) *FakeNetworkHost {
	host := &FakeNetworkHost{kubeClient: kubeClient, Legacy: true}
	return host
}

// GetPodByName returns empty pod struct
func (fnh *FakeNetworkHost) GetPodByName(name, namespace string) (*v1.Pod, bool) {
	return nil, false
}

// GetKubeClient returns nil for testing
func (fnh *FakeNetworkHost) GetKubeClient() clientset.Interface {
	return nil
}

// SupportsLegacyFeatures returns the Legacy value of FakeNetworkHost struct
func (fnh *FakeNetworkHost) SupportsLegacyFeatures() bool {
	return fnh.Legacy
}

type fakeNamespaceGetter struct {
	ns string
}

func (nh *fakeNamespaceGetter) GetNetNS(containerID string) (string, error) {
	return nh.ns, nil
}

// FakePortMappingGetter returns the record of port mapping for testing
type FakePortMappingGetter struct {
	PortMaps map[string][]*hostport.PortMapping
}

// GetPodPortMappings returns the port by containerID for testing
func (pm *FakePortMappingGetter) GetPodPortMappings(containerID string) ([]*hostport.PortMapping, error) {
	return pm.PortMaps[containerID], nil
}
