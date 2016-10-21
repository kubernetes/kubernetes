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
	internalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

type FakeCoreInternalVersion struct {
	*core.Fake
}

func (c *FakeCoreInternalVersion) ComponentStatuses() internalversion.ComponentStatusInterface {
	return &FakeComponentStatuses{c}
}

func (c *FakeCoreInternalVersion) ConfigMaps(namespace string) internalversion.ConfigMapInterface {
	return &FakeConfigMaps{c, namespace}
}

func (c *FakeCoreInternalVersion) Endpoints(namespace string) internalversion.EndpointsInterface {
	return &FakeEndpoints{c, namespace}
}

func (c *FakeCoreInternalVersion) Events(namespace string) internalversion.EventInterface {
	return &FakeEvents{c, namespace}
}

func (c *FakeCoreInternalVersion) LimitRanges(namespace string) internalversion.LimitRangeInterface {
	return &FakeLimitRanges{c, namespace}
}

func (c *FakeCoreInternalVersion) Namespaces() internalversion.NamespaceInterface {
	return &FakeNamespaces{c}
}

func (c *FakeCoreInternalVersion) Nodes() internalversion.NodeInterface {
	return &FakeNodes{c}
}

func (c *FakeCoreInternalVersion) PersistentVolumes() internalversion.PersistentVolumeInterface {
	return &FakePersistentVolumes{c}
}

func (c *FakeCoreInternalVersion) PersistentVolumeClaims(namespace string) internalversion.PersistentVolumeClaimInterface {
	return &FakePersistentVolumeClaims{c, namespace}
}

func (c *FakeCoreInternalVersion) Pods(namespace string) internalversion.PodInterface {
	return &FakePods{c, namespace}
}

func (c *FakeCoreInternalVersion) PodTemplates(namespace string) internalversion.PodTemplateInterface {
	return &FakePodTemplates{c, namespace}
}

func (c *FakeCoreInternalVersion) ReplicationControllers(namespace string) internalversion.ReplicationControllerInterface {
	return &FakeReplicationControllers{c, namespace}
}

func (c *FakeCoreInternalVersion) ResourceQuotas(namespace string) internalversion.ResourceQuotaInterface {
	return &FakeResourceQuotas{c, namespace}
}

func (c *FakeCoreInternalVersion) Secrets(namespace string) internalversion.SecretInterface {
	return &FakeSecrets{c, namespace}
}

func (c *FakeCoreInternalVersion) Services(namespace string) internalversion.ServiceInterface {
	return &FakeServices{c, namespace}
}

func (c *FakeCoreInternalVersion) ServiceAccounts(namespace string) internalversion.ServiceAccountInterface {
	return &FakeServiceAccounts{c, namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeCoreInternalVersion) RESTClient() restclient.Interface {
	var ret *restclient.RESTClient
	return ret
}
