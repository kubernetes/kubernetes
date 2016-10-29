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
	v1 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

type FakeAutoscalingV1 struct {
	*core.Fake
}

func (c *FakeAutoscalingV1) HorizontalPodAutoscalers(namespace string) v1.HorizontalPodAutoscalerInterface {
	return &FakeHorizontalPodAutoscalers{c, namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeAutoscalingV1) RESTClient() restclient.Interface {
	var ret *restclient.RESTClient
	return ret
}
