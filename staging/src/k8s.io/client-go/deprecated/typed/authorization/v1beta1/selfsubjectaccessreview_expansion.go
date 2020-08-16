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

package v1beta1

import (
	"context"

	authorizationapi "k8s.io/api/authorization/v1beta1"
)

type SelfSubjectAccessReviewExpansion interface {
	Create(sar *authorizationapi.SelfSubjectAccessReview) (result *authorizationapi.SelfSubjectAccessReview, err error)
	CreateContext(ctx context.Context, sar *authorizationapi.SelfSubjectAccessReview) (result *authorizationapi.SelfSubjectAccessReview, err error)
}

func (c *selfSubjectAccessReviews) Create(sar *authorizationapi.SelfSubjectAccessReview) (result *authorizationapi.SelfSubjectAccessReview, err error) {
	return c.CreateContext(context.Background(), sar)
}

func (c *selfSubjectAccessReviews) CreateContext(ctx context.Context, sar *authorizationapi.SelfSubjectAccessReview) (result *authorizationapi.SelfSubjectAccessReview, err error) {
	result = &authorizationapi.SelfSubjectAccessReview{}
	err = c.client.Post().
		Resource("selfsubjectaccessreviews").
		Body(sar).
		Do(ctx).
		Into(result)
	return
}
