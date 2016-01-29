/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	core "k8s.io/kubernetes/pkg/client/testing/core"
	unversioned "k8s.io/kubernetes/pkg/client/typed/generated/legacy/unversioned"
)

type FakeLegacy struct {
	*core.Fake
}

func (c *FakeLegacy) ComponentStatuses() unversioned.ComponentStatusInterface {
	return &FakeComponentStatuses{c}
}

func (c *FakeLegacy) Endpoints(namespace string) unversioned.EndpointsInterface {
	return &FakeEndpoints{c, namespace}
}

func (c *FakeLegacy) Events(namespace string) unversioned.EventInterface {
	return &FakeEvents{c, namespace}
}

func (c *FakeLegacy) LimitRanges(namespace string) unversioned.LimitRangeInterface {
	return &FakeLimitRanges{c, namespace}
}

func (c *FakeLegacy) Namespaces() unversioned.NamespaceInterface {
	return &FakeNamespaces{c}
}

func (c *FakeLegacy) Nodes() unversioned.NodeInterface {
	return &FakeNodes{c}
}

func (c *FakeLegacy) PersistentVolumes() unversioned.PersistentVolumeInterface {
	return &FakePersistentVolumes{c}
}

func (c *FakeLegacy) PersistentVolumeClaims(namespace string) unversioned.PersistentVolumeClaimInterface {
	return &FakePersistentVolumeClaims{c, namespace}
}

func (c *FakeLegacy) Pods(namespace string) unversioned.PodInterface {
	return &FakePods{c, namespace}
}

func (c *FakeLegacy) PodTemplates(namespace string) unversioned.PodTemplateInterface {
	return &FakePodTemplates{c, namespace}
}

func (c *FakeLegacy) ReplicationControllers(namespace string) unversioned.ReplicationControllerInterface {
	return &FakeReplicationControllers{c, namespace}
}

func (c *FakeLegacy) ResourceQuotas(namespace string) unversioned.ResourceQuotaInterface {
	return &FakeResourceQuotas{c, namespace}
}

func (c *FakeLegacy) Secrets(namespace string) unversioned.SecretInterface {
	return &FakeSecrets{c, namespace}
}

func (c *FakeLegacy) Services(namespace string) unversioned.ServiceInterface {
	return &FakeServices{c, namespace}
}

func (c *FakeLegacy) ServiceAccounts(namespace string) unversioned.ServiceAccountInterface {
	return &FakeServiceAccounts{c, namespace}
}
