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

package v1beta1

import (
	"context"

	authorizationapi "k8s.io/api/authorization/v1beta1"
)

type SelfSubjectRulesReviewExpansion interface {
	Create(srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error)
	CreateContext(ctx context.Context, srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error)
}

func (c *selfSubjectRulesReviews) Create(srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error) {
	return c.CreateContext(context.Background(), srr)
}

func (c *selfSubjectRulesReviews) CreateContext(ctx context.Context, srr *authorizationapi.SelfSubjectRulesReview) (result *authorizationapi.SelfSubjectRulesReview, err error) {
	result = &authorizationapi.SelfSubjectRulesReview{}
	err = c.client.Post().
		Resource("selfsubjectrulesreviews").
		Body(srr).
		Do(ctx).
		Into(result)
	return
}
