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
	v1alpha1 "k8s.io/kubernetes/pkg/client/typed/generated/controlplane/v1alpha1"
)

type FakeControlplane struct {
	*core.Fake
}

func (c *FakeControlplane) Clusters() v1alpha1.ClusterInterface {
	return &FakeClusters{c}
}
