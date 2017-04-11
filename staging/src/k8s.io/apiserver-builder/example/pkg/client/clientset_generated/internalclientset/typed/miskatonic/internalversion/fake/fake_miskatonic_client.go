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
	internalversion "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/internalclientset/typed/miskatonic/internalversion"
	rest "k8s.io/client-go/rest"
	testing "k8s.io/client-go/testing"
)

type FakeMiskatonic struct {
	*testing.Fake
}

func (c *FakeMiskatonic) Scales(namespace string) internalversion.ScaleInterface {
	return &FakeScales{c, namespace}
}

func (c *FakeMiskatonic) Universities(namespace string) internalversion.UniversityInterface {
	return &FakeUniversities{c, namespace}
}

func (c *FakeMiskatonic) UniversitySpecs(namespace string) internalversion.UniversitySpecInterface {
	return &FakeUniversitySpecs{c, namespace}
}

func (c *FakeMiskatonic) UniversityStatuses(namespace string) internalversion.UniversityStatusInterface {
	return &FakeUniversityStatuses{c, namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeMiskatonic) RESTClient() rest.Interface {
	var ret *rest.RESTClient
	return ret
}
