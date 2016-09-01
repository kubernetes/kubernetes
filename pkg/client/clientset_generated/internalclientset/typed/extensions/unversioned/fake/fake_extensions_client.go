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
	unversioned "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

type FakeExtensions struct {
	*core.Fake
}

func (c *FakeExtensions) DaemonSets(namespace string) unversioned.DaemonSetInterface {
	return &FakeDaemonSets{c, namespace}
}

func (c *FakeExtensions) Deployments(namespace string) unversioned.DeploymentInterface {
	return &FakeDeployments{c, namespace}
}

func (c *FakeExtensions) Ingresses(namespace string) unversioned.IngressInterface {
	return &FakeIngresses{c, namespace}
}

func (c *FakeExtensions) PodSecurityPolicies() unversioned.PodSecurityPolicyInterface {
	return &FakePodSecurityPolicies{c}
}

func (c *FakeExtensions) ReplicaSets(namespace string) unversioned.ReplicaSetInterface {
	return &FakeReplicaSets{c, namespace}
}

func (c *FakeExtensions) Scales(namespace string) unversioned.ScaleInterface {
	return &FakeScales{c, namespace}
}

func (c *FakeExtensions) ThirdPartyResources() unversioned.ThirdPartyResourceInterface {
	return &FakeThirdPartyResources{c}
}

// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeExtensions) GetRESTClient() *restclient.RESTClient {
	return nil
}
