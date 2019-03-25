/*
Copyright 2018 The Kubernetes Authors.

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

package e2e_kubeadm

import (
	authv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
)

// ConfigMaps utils

func GetConfigMap(c clientset.Interface, namespace, name string) *corev1.ConfigMap {
	r, err := c.CoreV1().
		ConfigMaps(namespace).
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ConfigMap %q from namespace %q", name, namespace)
	return r
}

// RBAC utils

func ExpectRole(c clientset.Interface, namespace, name string) {
	_, err := c.RbacV1().
		Roles(namespace).
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting Role %q from namespace %q", name, namespace)
}

func ExpectRoleBinding(c clientset.Interface, namespace, name string) {
	_, err := c.RbacV1().
		RoleBindings(namespace).
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting RoleBinding %q from namespace %q", name, namespace)
}

func ExpectClusterRole(c clientset.Interface, name string) {
	_, err := c.RbacV1().
		ClusterRoles().
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRole %q", name)
}

func ExpectClusterRoleBinding(c clientset.Interface, name string) {
	_, err := c.RbacV1().
		ClusterRoleBindings().
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRoleBindings %q", name)
}

func ExpectClusterRoleBindingWithSubjectAndRole(c clientset.Interface, name, subjectKind, subject, role string) {
	binding, err := c.RbacV1().
		ClusterRoleBindings().
		Get(name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRoleBindings %q", name)
	Expect(binding.Subjects).To(
		ContainElement(subjectMatcher(
			subject,
			subjectKind,
		)),
		"ClusterRole %q does not have %s %q as subject", name, subjectKind, subject,
	)
	Expect(binding.RoleRef.Name).To(
		Equal(role),
		"ClusterRole %q does not have %q as role", name, role,
	)
}

func ExpectSubjectHasAccessToResource(c clientset.Interface, subjectKind, subject string, resource *authv1.ResourceAttributes) {
	var sar *authv1.SubjectAccessReview
	switch subjectKind {
	case rbacv1.GroupKind:
		sar = &authv1.SubjectAccessReview{
			Spec: authv1.SubjectAccessReviewSpec{
				Groups:             []string{subject},
				ResourceAttributes: resource,
			},
		}
	case rbacv1.UserKind:
		fallthrough
	case rbacv1.ServiceAccountKind:
		sar = &authv1.SubjectAccessReview{
			Spec: authv1.SubjectAccessReviewSpec{
				User:               subject,
				ResourceAttributes: resource,
			},
		}
	default:
		framework.Failf("invalid subjectKind %s", subjectKind)
	}

	s, err := c.AuthorizationV1().SubjectAccessReviews().Create(sar)
	framework.ExpectNoError(err, "error getting SubjectAccessReview for %s %s to resource %+v", subjectKind, subject, *sar.Spec.ResourceAttributes)

	Expect(s.Status.Allowed).Should(BeTrue(), "%s %s has no access to resource %+v", subjectKind, subject, *sar.Spec.ResourceAttributes)
}

// matchers

func subjectMatcher(name, kind string) OmegaMatcher {
	return gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Name": Equal(name),
		"Kind": Equal(kind),
	})
}
