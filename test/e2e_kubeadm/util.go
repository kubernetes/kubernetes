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

package kubeadm

import (
	"context"

	appsv1 "k8s.io/api/apps/v1"
	authv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
)

// ServiceAccounts utils

// ExpectServiceAccount expects to be able to get the ServiceAccount with specific name from the namespace
func ExpectServiceAccount(c clientset.Interface, namespace, name string) {
	_, err := c.CoreV1().
		ServiceAccounts(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ServiceAccount %q from namespace %q", name, namespace)
}

// Secret utils

// GetSecret gets Secret with specific name from the namespace
func GetSecret(c clientset.Interface, namespace, name string) *corev1.Secret {
	r, err := c.CoreV1().
		Secrets(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting Secret %q from namespace %q", name, namespace)
	return r
}

// ConfigMaps utils

// GetConfigMap gets ConfigMap with specific name from the namespace
func GetConfigMap(c clientset.Interface, namespace, name string) *corev1.ConfigMap {
	r, err := c.CoreV1().
		ConfigMaps(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ConfigMap %q from namespace %q", name, namespace)
	return r
}

// Service utils

// ExpectService expects to be able to get the Service with specific name from the namespace
func ExpectService(c clientset.Interface, namespace, name string) {
	_, err := c.CoreV1().
		Services(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting Service %q from namespace %q", name, namespace)
}

// Deployments utils

// GetDeployment gets Deployment with specific name from the namespace
func GetDeployment(c clientset.Interface, namespace, name string) *appsv1.Deployment {
	r, err := c.AppsV1().
		Deployments(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting Deployment %q from namespace %q", name, namespace)
	return r
}

// DaemonSets utils

// GetDaemonSet gets DaemonSet with specific name from the namespace
func GetDaemonSet(c clientset.Interface, namespace, name string) *appsv1.DaemonSet {
	r, err := c.AppsV1().
		DaemonSets(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting DaemonSet %q from namespace %q", name, namespace)
	return r
}

// RBAC utils

// ExpectRole expects to be able to get the Role with specific name from the namespace
func ExpectRole(c clientset.Interface, namespace, name string) {
	_, err := c.RbacV1().
		Roles(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting Role %q from namespace %q", name, namespace)
}

// ExpectRoleBinding expects to be able to get the RoleBinding with specific name from the namespace
func ExpectRoleBinding(c clientset.Interface, namespace, name string) {
	_, err := c.RbacV1().
		RoleBindings(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting RoleBinding %q from namespace %q", name, namespace)
}

// ExpectClusterRole expects to be able to get the ClusterRole with specific name
func ExpectClusterRole(c clientset.Interface, name string) {
	_, err := c.RbacV1().
		ClusterRoles().
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRole %q", name)
}

// ExpectClusterRoleBinding expects to be able to get the ClusterRoleBinding with specific name
func ExpectClusterRoleBinding(c clientset.Interface, name string) {
	_, err := c.RbacV1().
		ClusterRoleBindings().
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRoleBindings %q", name)
}

// ExpectClusterRoleBindingWithSubjectAndRole expects to be able to get the ClusterRoleBinding with specific name, subject and role
func ExpectClusterRoleBindingWithSubjectAndRole(c clientset.Interface, name, subjectKind, subject, role string) {
	binding, err := c.RbacV1().
		ClusterRoleBindings().
		Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting ClusterRoleBindings %q", name)
	gomega.Expect(binding.Subjects).To(
		gomega.ContainElement(subjectMatcher(
			subject,
			subjectKind,
		)),
		"ClusterRole %q does not have %s %q as subject", name, subjectKind, subject,
	)
	gomega.Expect(binding.RoleRef.Name).To(
		gomega.Equal(role),
		"ClusterRole %q does not have %q as role", name, role,
	)
}

// ExpectSubjectHasAccessToResource expects that the subject has access to the target resource
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

	s, err := c.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	framework.ExpectNoError(err, "error getting SubjectAccessReview for %s %s to resource %+v", subjectKind, subject, *sar.Spec.ResourceAttributes)

	gomega.Expect(s.Status.Allowed).Should(gomega.BeTrueBecause("%s %s has no access to resource %+v", subjectKind, subject, *sar.Spec.ResourceAttributes))
}

// matchers

func subjectMatcher(name, kind string) gomega.OmegaMatcher {
	return gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Name": gomega.Equal(name),
		"Kind": gomega.Equal(kind),
	})
}
