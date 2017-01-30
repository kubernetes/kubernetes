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
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication/v1"
)

type TokenReviewExpansion interface {
	Create(tokenReview *authenticationapi.TokenReview) (result *authenticationapi.TokenReview, err error)
}

func (c *tokenReviews) Create(tokenReview *authenticationapi.TokenReview) (result *authenticationapi.TokenReview, err error) {
	result = &authenticationapi.TokenReview{}
	err = c.client.Post().
		Resource("tokenreviews").
		Body(tokenReview).
		Do().
		Into(result)
	return
}
