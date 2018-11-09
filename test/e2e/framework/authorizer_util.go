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
	"k8s.io/klog"
	"sync"
	"time"

	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	v1beta1authorization "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"
	v1beta1rbac "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
)

const (
	policyCachePollInterval = 100 * time.Millisecond
	policyCachePollTimeout  = 5 * time.Second
)

// WaitForAuthorizationUpdate checks if the given user can perform the named verb and action.
// If policyCachePollTimeout is reached without the expected condition matching, an error is returned
func WaitForAuthorizationUpdate(c v1beta1authorization.SubjectAccessReviewsGetter, user, namespace, verb string, resource schema.GroupResource, allowed bool) error {
	return WaitForNamedAuthorizationUpdate(c, user, namespace, verb, "", resource, allowed)
}

// WaitForAuthorizationUpdate checks if the given user can perform the named verb and action on the named resource.
// If policyCachePollTimeout is reached without the expected condition matching, an error is returned
func WaitForNamedAuthorizationUpdate(c v1beta1authorization.SubjectAccessReviewsGetter, user, namespace, verb, resourceName string, resource schema.GroupResource, allowed bool) error {
	review := &authorizationv1beta1.SubjectAccessReview{
		Spec: authorizationv1beta1.SubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1beta1.ResourceAttributes{
				Group:     resource.Group,
				Verb:      verb,
				Resource:  resource.Resource,
				Namespace: namespace,
				Name:      resourceName,
			},
			User: user,
		},
	}
	err := wait.Poll(policyCachePollInterval, policyCachePollTimeout, func() (bool, error) {
		response, err := c.SubjectAccessReviews().Create(review)
		// GKE doesn't enable the SAR endpoint.  Without this endpoint, we cannot determine if the policy engine
		// has adjusted as expected.  In this case, simply wait one second and hope it's up to date
		if apierrors.IsNotFound(err) {
			klog.Info("SubjectAccessReview endpoint is missing")
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

// BindClusterRole binds the cluster role at the cluster scope
func BindClusterRole(c v1beta1rbac.ClusterRoleBindingsGetter, clusterRole, ns string, subjects ...rbacv1beta1.Subject) {
	// Since the namespace names are unique, we can leave this lying around so we don't have to race any caches
	_, err := c.ClusterRoleBindings().Create(&rbacv1beta1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns + "--" + clusterRole,
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     clusterRole,
		},
		Subjects: subjects,
	})

	// if we failed, don't fail the entire test because it may still work. RBAC may simply be disabled.
	if err != nil {
		klog.Errorf("Error binding clusterrole/%s for %q for %v\n", clusterRole, ns, subjects)
	}
}

// BindClusterRoleInNamespace binds the cluster role at the namespace scope
func BindClusterRoleInNamespace(c v1beta1rbac.RoleBindingsGetter, clusterRole, ns string, subjects ...rbacv1beta1.Subject) {
	bindInNamespace(c, "ClusterRole", clusterRole, ns, subjects...)
}

// BindRoleInNamespace binds the role at the namespace scope
func BindRoleInNamespace(c v1beta1rbac.RoleBindingsGetter, role, ns string, subjects ...rbacv1beta1.Subject) {
	bindInNamespace(c, "Role", role, ns, subjects...)
}

func bindInNamespace(c v1beta1rbac.RoleBindingsGetter, roleType, role, ns string, subjects ...rbacv1beta1.Subject) {
	// Since the namespace names are unique, we can leave this lying around so we don't have to race any caches
	_, err := c.RoleBindings(ns).Create(&rbacv1beta1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns + "--" + role,
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     roleType,
			Name:     role,
		},
		Subjects: subjects,
	})

	// if we failed, don't fail the entire test because it may still work. RBAC may simply be disabled.
	if err != nil {
		klog.Errorf("Error binding %s/%s into %q for %v\n", roleType, role, ns, subjects)
	}
}

var (
	isRBACEnabledOnce sync.Once
	isRBACEnabled     bool
)

func IsRBACEnabled(f *Framework) bool {
	isRBACEnabledOnce.Do(func() {
		crs, err := f.ClientSet.RbacV1().ClusterRoles().List(metav1.ListOptions{})
		if err != nil {
			Logf("Error listing ClusterRoles; assuming RBAC is disabled: %v", err)
			isRBACEnabled = false
		} else if crs == nil || len(crs.Items) == 0 {
			Logf("No ClusterRoles found; assuming RBAC is disabled.")
			isRBACEnabled = false
		} else {
			Logf("Found ClusterRoles; assuming RBAC is enabled.")
			isRBACEnabled = true
		}
	})
	return isRBACEnabled
}
