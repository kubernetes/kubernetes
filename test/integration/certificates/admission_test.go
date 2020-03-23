/*
Copyright 2020 The Kubernetes Authors.

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

package certificates

import (
	"context"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	v1authorization "k8s.io/client-go/kubernetes/typed/authorization/v1"
)

// waitForNamedAuthorizationUpdate checks if the given user can perform the named verb and action on the named resource.
// Copied from k8s.io/kubernetes/test/e2e/framework/auth.
func waitForNamedAuthorizationUpdate(t *testing.T, c v1authorization.SubjectAccessReviewsGetter, user, namespace, verb, resourceName string, resource schema.GroupResource, allowed bool) {
	review := &authorizationv1.SubjectAccessReview{
		Spec: authorizationv1.SubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Group:     resource.Group,
				Verb:      verb,
				Resource:  resource.Resource,
				Namespace: namespace,
				Name:      resourceName,
			},
			User: user,
		},
	}

	if err := wait.Poll(time.Millisecond*100, time.Second*5, func() (bool, error) {
		response, err := c.SubjectAccessReviews().Create(context.TODO(), review, metav1.CreateOptions{})
		if err != nil {
			return false, err
		}
		if response.Status.Allowed != allowed {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}
