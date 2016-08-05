/*
Copyright 2015 The Kubernetes Authors.

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

package unversioned

import (
	"k8s.io/kubernetes/pkg/apis/authorization"
)

type SubjectAccessReviewsInterface interface {
	SubjectAccessReviews() SubjectAccessReviewInterface
}

type SubjectAccessReviewInterface interface {
	Create(subjectAccessReview *authorization.SubjectAccessReview) (*authorization.SubjectAccessReview, error)
}

type subjectAccessReviews struct {
	client *AuthorizationClient
}

func newSubjectAccessReviews(c *AuthorizationClient) *subjectAccessReviews {
	return &subjectAccessReviews{
		client: c,
	}
}

func (c *subjectAccessReviews) Create(subjectAccessReview *authorization.SubjectAccessReview) (result *authorization.SubjectAccessReview, err error) {
	result = &authorization.SubjectAccessReview{}
	err = c.client.Post().Resource("subjectAccessReviews").Body(subjectAccessReview).Do().Into(result)
	return
}
