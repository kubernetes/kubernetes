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
	"context"

	authorizationapi "k8s.io/api/authorization/v1beta1"
	core "k8s.io/client-go/testing"
)

func (c *FakeSelfSubjectRulesReviews) Create(srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error) {
	return c.CreateContext(context.Background(), srr)
}

func (c *FakeSelfSubjectRulesReviews) CreateContext(ctx context.Context, srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error) {
	obj, err := c.Fake.Invokes(core.NewRootCreateAction(authorizationapi.SchemeGroupVersion.WithResource("selfsubjectrulesreviews"), srr), &authorizationapi.SelfSubjectRulesReview{})
	return obj.(*authorizationapi.SelfSubjectRulesReview), err
}
