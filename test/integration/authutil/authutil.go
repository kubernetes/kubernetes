/*
Copyright 2021 The Kubernetes Authors.

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

package authutil

import (
	"context"
	"strings"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
)

// WaitForNamedAuthorizationUpdate checks if the given user can perform the named verb and action on the named resource.
// Copied from k8s.io/kubernetes/test/e2e/framework/auth.
func WaitForNamedAuthorizationUpdate(t *testing.T, ctx context.Context, c authorizationv1client.SubjectAccessReviewsGetter, user, namespace, verb, resourceName string, resource schema.GroupResource, allowed bool) {
	t.Helper()

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

	if err := wait.Poll(200*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		response, err := c.SubjectAccessReviews().Create(ctx, review, metav1.CreateOptions{})
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

func GrantUserAuthorization(t *testing.T, ctx context.Context, adminClient clientset.Interface, username string, rule rbacv1.PolicyRule) {
	grantAuthorization(t, ctx, adminClient, username, "", rbacv1.UserKind, rule)
}

func GrantServiceAccountAuthorization(t *testing.T, ctx context.Context, adminClient clientset.Interface, serviceAccountName, serviceAccountNamespace string, rule rbacv1.PolicyRule) {
	grantAuthorization(t, ctx, adminClient, serviceAccountName, serviceAccountNamespace, rbacv1.ServiceAccountKind, rule)
}

func grantAuthorization(t *testing.T, ctx context.Context, adminClient clientset.Interface, name, namespace, accountKind string, rule rbacv1.PolicyRule) {
	t.Helper()

	cr, err := adminClient.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{GenerateName: strings.Replace(t.Name(), "/", "--", -1)},
		Rules: []rbacv1.PolicyRule{
			rule,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = adminClient.RbacV1().ClusterRoles().Delete(ctx, cr.Name, metav1.DeleteOptions{})
	})

	crb, err := adminClient.RbacV1().ClusterRoleBindings().Create(ctx, &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{GenerateName: strings.Replace(t.Name(), "/", "--", -1)},
		Subjects: []rbacv1.Subject{
			{
				Kind: accountKind,
				// APIGroup defaults to the appropriate value for both users and service accounts
				Name:      name,
				Namespace: namespace,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     cr.Name,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = adminClient.RbacV1().ClusterRoleBindings().Delete(ctx, crb.Name, metav1.DeleteOptions{})
	})

	var resourceName string
	if len(rule.ResourceNames) > 0 {
		resourceName = rule.ResourceNames[0]
	}

	subjectName := name
	if accountKind == rbacv1.ServiceAccountKind {
		subjectName = "system:serviceaccount:" + namespace + ":" + name
	}

	WaitForNamedAuthorizationUpdate(
		t,
		ctx,
		adminClient.AuthorizationV1(),
		subjectName,
		namespace,
		rule.Verbs[0],
		resourceName,
		schema.GroupResource{Group: rule.APIGroups[0], Resource: rule.Resources[0]},
		true,
	)
}
