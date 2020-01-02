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

package v1

import (
	"context"

	authorizationapi "k8s.io/api/authorization/v1"
)

// The SubjectAccessReviewExpansion interface allows manually adding extra methods to the AuthorizationInterface.
type SubjectAccessReviewExpansion interface {
	Create(sar *authorizationapi.SubjectAccessReview) (result *authorizationapi.SubjectAccessReview, err error)
	CreateContext(ctx context.Context, sar *authorizationapi.SubjectAccessReview) (result *authorizationapi.SubjectAccessReview, err error)
}

func (c *subjectAccessReviews) Create(sar *authorizationapi.SubjectAccessReview) (result *authorizationapi.SubjectAccessReview, err error) {
	return c.CreateContext(context.Background(), sar)
}

func (c *subjectAccessReviews) CreateContext(ctx context.Context, sar *authorizationapi.SubjectAccessReview) (result *authorizationapi.SubjectAccessReview, err error) {
	result = &authorizationapi.SubjectAccessReview{}
	err = c.client.Post().
		Context(ctx).
		Resource("subjectaccessreviews").
		Body(sar).
		Do().
		Into(result)
	return
}
