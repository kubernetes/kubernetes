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

package framework

import (
	"time"

	apierrors "k8s.io/kubernetes/pkg/api/errors"
	authorizationv1beta1 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1"
	v1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1beta1"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	policyCachePollInterval = 100 * time.Millisecond
	policyCachePollTimeout  = 5 * time.Second
)

// WaitForAuthorizationUpdate checks if the given user can perform the named verb and action.
// If policyCachePollTimeout is reached without the expected condition matching, an error is returned
func WaitForAuthorizationUpdate(c v1beta1authorization.SubjectAccessReviewsGetter, user, namespace, verb string, resource schema.GroupResource, allowed bool) error {
	review := &authorizationv1beta1.SubjectAccessReview{
		Spec: authorizationv1beta1.SubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1beta1.ResourceAttributes{
				Group:     resource.Group,
				Verb:      verb,
				Resource:  resource.Resource,
				Namespace: namespace,
			},
			User: user,
		},
	}
	err := wait.Poll(policyCachePollInterval, policyCachePollTimeout, func() (bool, error) {
		response, err := c.SubjectAccessReviews().Create(review)
		// GKE doesn't enable the SAR endpoint.  Without this endpoint, we cannot determine if the policy engine
		// has adjusted as expected.  In this case, simply wait one second and hope it's up to date
		if apierrors.IsNotFound(err) {
			time.Sleep(1 * time.Second)
			return true, nil
		}
		if err != nil {
			return false, err
		}
		if response.Status.Allowed != allowed {
			return false, nil
		}
		return true, nil
	})
	return err
}
