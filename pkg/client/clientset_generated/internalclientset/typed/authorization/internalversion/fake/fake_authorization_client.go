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
	internalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

type FakeAuthorization struct {
	*core.Fake
}

func (c *FakeAuthorization) LocalSubjectAccessReviews(namespace string) internalversion.LocalSubjectAccessReviewInterface {
	return &FakeLocalSubjectAccessReviews{c, namespace}
}

func (c *FakeAuthorization) SelfSubjectAccessReviews() internalversion.SelfSubjectAccessReviewInterface {
	return &FakeSelfSubjectAccessReviews{c}
}

func (c *FakeAuthorization) SubjectAccessReviews() internalversion.SubjectAccessReviewInterface {
	return &FakeSubjectAccessReviews{c}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeAuthorization) RESTClient() restclient.Interface {
	var ret *restclient.RESTClient
	return ret
}
