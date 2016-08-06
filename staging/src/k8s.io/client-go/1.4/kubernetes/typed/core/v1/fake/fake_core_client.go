/*
Copyright 2016 The Kubernetes Authors.

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

package fake

import (
	v1 "k8s.io/client-go/1.4/kubernetes/typed/core/v1"
	rest "k8s.io/client-go/1.4/rest"
	testing "k8s.io/client-go/1.4/testing"
)

type FakeCore struct {
	*testing.Fake
}

func (c *FakeCore) ComponentStatuses() v1.ComponentStatusInterface {
	return &FakeComponentStatuses{c}
}

func (c *FakeCore) ConfigMaps(namespace string) v1.ConfigMapInterface {
	return &FakeConfigMaps{c, namespace}
}

func (c *FakeCore) Endpoints(namespace string) v1.EndpointsInterface {
	return &FakeEndpoints{c, namespace}
}

func (c *FakeCore) Events(namespace string) v1.EventInterface {
	return &FakeEvents{c, namespace}
}

func (c *FakeCore) LimitRanges(namespace string) v1.LimitRangeInterface {
	return &FakeLimitRanges{c, namespace}
}

func (c *FakeCore) Namespaces() v1.NamespaceInterface {
	return &FakeNamespaces{c}
}

func (c *FakeCore) Nodes() v1.NodeInterface {
	return &FakeNodes{c}
}

func (c *FakeCore) PersistentVolumes() v1.PersistentVolumeInterface {
	return &FakePersistentVolumes{c}
}

func (c *FakeCore) Pods(namespace string) v1.PodInterface {
	return &FakePods{c, namespace}
}

func (c *FakeCore) PodTemplates(namespace string) v1.PodTemplateInterface {
	return &FakePodTemplates{c, namespace}
}

func (c *FakeCore) ReplicationControllers(namespace string) v1.ReplicationControllerInterface {
	return &FakeReplicationControllers{c, namespace}
}

func (c *FakeCore) ResourceQuotas(namespace string) v1.ResourceQuotaInterface {
	return &FakeResourceQuotas{c, namespace}
}

func (c *FakeCore) Secrets(namespace string) v1.SecretInterface {
	return &FakeSecrets{c, namespace}
}

func (c *FakeCore) Services(namespace string) v1.ServiceInterface {
	return &FakeServices{c, namespace}
}

func (c *FakeCore) ServiceAccounts(namespace string) v1.ServiceAccountInterface {
	return &FakeServiceAccounts{c, namespace}
}

// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeCore) GetRESTClient() *rest.RESTClient {
	return nil
}
