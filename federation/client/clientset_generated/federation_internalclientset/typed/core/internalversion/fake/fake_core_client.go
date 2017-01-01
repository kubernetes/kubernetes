/*
Copyright 2017 The Kubernetes Authors.

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
	internalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

type FakeCore struct {
	*core.Fake
}

func (c *FakeCore) ConfigMaps(namespace string) internalversion.ConfigMapInterface {
	return &FakeConfigMaps{c, namespace}
}

func (c *FakeCore) Events(namespace string) internalversion.EventInterface {
	return &FakeEvents{c, namespace}
}

func (c *FakeCore) Namespaces() internalversion.NamespaceInterface {
	return &FakeNamespaces{c}
}

func (c *FakeCore) Secrets(namespace string) internalversion.SecretInterface {
	return &FakeSecrets{c, namespace}
}

func (c *FakeCore) Services(namespace string) internalversion.ServiceInterface {
	return &FakeServices{c, namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeCore) RESTClient() restclient.Interface {
	var ret *restclient.RESTClient
	return ret
}
