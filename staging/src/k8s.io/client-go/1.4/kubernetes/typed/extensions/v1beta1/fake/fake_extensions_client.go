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
	v1beta1 "k8s.io/client-go/1.4/kubernetes/typed/extensions/v1beta1"
	rest "k8s.io/client-go/1.4/rest"
	testing "k8s.io/client-go/1.4/testing"
)

type FakeExtensions struct {
	*testing.Fake
}

func (c *FakeExtensions) DaemonSets(namespace string) v1beta1.DaemonSetInterface {
	return &FakeDaemonSets{c, namespace}
}

func (c *FakeExtensions) Deployments(namespace string) v1beta1.DeploymentInterface {
	return &FakeDeployments{c, namespace}
}

func (c *FakeExtensions) Ingresses(namespace string) v1beta1.IngressInterface {
	return &FakeIngresses{c, namespace}
}

func (c *FakeExtensions) Jobs(namespace string) v1beta1.JobInterface {
	return &FakeJobs{c, namespace}
}

func (c *FakeExtensions) PodSecurityPolicies() v1beta1.PodSecurityPolicyInterface {
	return &FakePodSecurityPolicies{c}
}

func (c *FakeExtensions) ReplicaSets(namespace string) v1beta1.ReplicaSetInterface {
	return &FakeReplicaSets{c, namespace}
}

func (c *FakeExtensions) Scales(namespace string) v1beta1.ScaleInterface {
	return &FakeScales{c, namespace}
}

func (c *FakeExtensions) StorageClasses() v1beta1.StorageClassInterface {
	return &FakeStorageClasses{c}
}

func (c *FakeExtensions) ThirdPartyResources() v1beta1.ThirdPartyResourceInterface {
	return &FakeThirdPartyResources{c}
}

// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeExtensions) GetRESTClient() *rest.RESTClient {
	return nil
}
