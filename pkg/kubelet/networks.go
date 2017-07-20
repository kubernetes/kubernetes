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

package kubelet

import (
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
)

// This just exports required functions from kubelet proper, for use by network
// plugins.
// TODO(#35457): get rid of this backchannel to the kubelet. The scope of
// the back channel is restricted to host-ports/testing, and restricted
// to kubenet. No other network plugin wrapper needs it. Other plugins
// only require a way to access namespace information, which they can do
// directly through the methods implemented by criNetworkHost.
type networkHost struct {
	kubelet *Kubelet
}

func (nh *networkHost) GetPodByName(name, namespace string) (*v1.Pod, bool) {
	return nh.kubelet.GetPodByName(name, namespace)
}

func (nh *networkHost) GetKubeClient() clientset.Interface {
	return nh.kubelet.kubeClient
}

func (nh *networkHost) GetRuntime() kubecontainer.Runtime {
	return nh.kubelet.GetRuntime()
}

func (nh *networkHost) SupportsLegacyFeatures() bool {
	return true
}

// criNetworkHost implements the part of network.Host required by the
// cri (NamespaceGetter). It leechs off networkHost for all other
// methods, because networkHost is slated for deletion.
type criNetworkHost struct {
	*networkHost
	// criNetworkHost currently support legacy features. Hence no need to support PortMappingGetter
	*network.NoopPortMappingGetter
}

// GetNetNS returns the network namespace of the given containerID.
// This method satisfies the network.NamespaceGetter interface for
// networkHost. It's only meant to be used from network plugins
// that are directly invoked by the kubelet (aka: legacy, pre-cri).
// Any network plugin invoked by a cri must implement NamespaceGetter
// to talk directly to the runtime instead.
func (c *criNetworkHost) GetNetNS(containerID string) (string, error) {
	return c.kubelet.GetRuntime().GetNetNS(kubecontainer.ContainerID{Type: "", ID: containerID})
}

// NoOpLegacyHost implements the network.LegacyHost interface for the remote
// runtime shim by just returning empties. It doesn't support legacy features
// like host port and bandwidth shaping.
type NoOpLegacyHost struct{}

// GetPodByName always returns "nil, true" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetPodByName(namespace, name string) (*v1.Pod, bool) {
	return nil, true
}

// GetKubeClient always returns "nil" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetKubeClient() clientset.Interface {
	return nil
}

// GetRuntime always returns "nil" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetRuntime() kubecontainer.Runtime {
	return nil
}

// SupportsLegacyFeatures always returns "false" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) SupportsLegacyFeatures() bool {
	return false
}
