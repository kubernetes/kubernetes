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

package auth

import (
	"sync"
	"time"

	"github.com/pkg/errors"
	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	v1authorization "k8s.io/client-go/kubernetes/typed/authorization/v1"
	v1rbac "k8s.io/client-go/kubernetes/typed/rbac/v1"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

const (
	policyCachePollInterval = 100 * time.Millisecond
	policyCachePollTimeout  = 5 * time.Second
)

type bindingsGetter interface {
	v1rbac.RoleBindingsGetter
	v1rbac.ClusterRoleBindingsGetter
	v1rbac.ClusterRolesGetter
}

// WaitForAuthorizationUpdate checks if the given user can perform the named verb and action.
// If policyCachePollTimeout is reached without the expected condition matching, an error is returned
func WaitForAuthorizationUpdate(c v1authorization.SubjectAccessReviewsGetter, user, namespace, verb string, resource schema.GroupResource, allowed bool) error {
	return WaitForNamedAuthorizationUpdate(c, user, namespace, verb, "", resource, allowed)
}

// WaitForNamedAuthorizationUpdate checks if the given user can perform the named verb and action on the named resource.
// If policyCachePollTimeout is reached without the expected condition matching, an error is returned
func WaitForNamedAuthorizationUpdate(c v1authorization.SubjectAccessReviewsGetter, user, namespace, verb, resourceName string, resource schema.GroupResource, allowed bool) error {
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

	err := wait.Poll(policyCachePollInterval, policyCachePollTimeout, func() (bool, error) {
		response, err := c.SubjectAccessReviews().Create(review)
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

// BindClusterRole binds the cluster role at the cluster scope. If RBAC is not enabled, nil
// is returned with no action.
func BindClusterRole(c bindingsGetter, clusterRole, ns string, subjects ...rbacv1.Subject) error {
	if !IsRBACEnabled(c) {
		return nil
	}

	// Since the namespace names are unique, we can leave this lying around so we don't have to race any caches
	_, err := c.ClusterRoleBindings().Create(&rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns + "--" + clusterRole,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     clusterRole,
		},
		Subjects: subjects,
	})

	if err != nil {
		return errors.Wrapf(err, "binding clusterrole/%s for %q for %v", clusterRole, ns, subjects)
	}

	return nil
}

// BindClusterRoleInNamespace binds the cluster role at the namespace scope. If RBAC is not enabled, nil
// is returned with no action.
func BindClusterRoleInNamespace(c bindingsGetter, clusterRole, ns string, subjects ...rbacv1.Subject) error {
	return bindInNamespace(c, "ClusterRole", clusterRole, ns, subjects...)
}

// BindRoleInNamespace binds the role at the namespace scope. If RBAC is not enabled, nil
// is returned with no action.
func BindRoleInNamespace(c bindingsGetter, role, ns string, subjects ...rbacv1.Subject) error {
	return bindInNamespace(c, "Role", role, ns, subjects...)
}

func bindInNamespace(c bindingsGetter, roleType, role, ns string, subjects ...rbacv1.Subject) error {
	if !IsRBACEnabled(c) {
		return nil
	}

	// Since the namespace names are unique, we can leave this lying around so we don't have to race any caches
	_, err := c.RoleBindings(ns).Create(&rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns + "--" + role,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     roleType,
			Name:     role,
		},
		Subjects: subjects,
	})

	if err != nil {
		return errors.Wrapf(err, "binding %s/%s into %q for %v", roleType, role, ns, subjects)
	}

	return nil
}

var (
	isRBACEnabledOnce sync.Once
	isRBACEnabled     bool
)

// IsRBACEnabled returns true if RBAC is enabled. Otherwise false.
func IsRBACEnabled(crGetter v1rbac.ClusterRolesGetter) bool {
	isRBACEnabledOnce.Do(func() {
		crs, err := crGetter.ClusterRoles().List(metav1.ListOptions{})
		if err != nil {
			e2elog.Logf("Error listing ClusterRoles; assuming RBAC is disabled: %v", err)
			isRBACEnabled = false
		} else if crs == nil || len(crs.Items) == 0 {
			e2elog.Logf("No ClusterRoles found; assuming RBAC is disabled.")
			isRBACEnabled = false
		} else {
			e2elog.Logf("Found ClusterRoles; assuming RBAC is enabled.")
			isRBACEnabled = true
		}
	})

	return isRBACEnabled
}
