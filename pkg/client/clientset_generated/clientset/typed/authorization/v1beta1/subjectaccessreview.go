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
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// SubjectAccessReviewsGetter has a method to return a SubjectAccessReviewInterface.
// A group's client should implement this interface.
type SubjectAccessReviewsGetter interface {
	SubjectAccessReviews() SubjectAccessReviewInterface
}

// SubjectAccessReviewInterface has methods to work with SubjectAccessReview resources.
type SubjectAccessReviewInterface interface {
	SubjectAccessReviewExpansion
}

// subjectAccessReviews implements SubjectAccessReviewInterface
type subjectAccessReviews struct {
	client restclient.Interface
}

// newSubjectAccessReviews returns a SubjectAccessReviews
func newSubjectAccessReviews(c *AuthorizationV1beta1Client) *subjectAccessReviews {
	return &subjectAccessReviews{
		client: c.RESTClient(),
	}
}
