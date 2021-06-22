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

package framework

import (
	"context"
	"fmt"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
)

const (
	podSecurityPolicyPrivileged = "e2e-test-privileged-psp"

	// allowAny is the wildcard used to allow any profile.
	allowAny = "*"

	// allowedProfilesAnnotationKey specifies the allowed seccomp profiles.
	allowedProfilesAnnotationKey = "seccomp.security.alpha.kubernetes.io/allowedProfileNames"
)

var (
	isPSPEnabledOnce sync.Once
	isPSPEnabled     bool
)

// privilegedPSP creates a PodSecurityPolicy that allows everything.
func privilegedPSP(name string) *policyv1beta1.PodSecurityPolicy {
	allowPrivilegeEscalation := true
	return &policyv1beta1.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{allowedProfilesAnnotationKey: allowAny},
		},
		Spec: policyv1beta1.PodSecurityPolicySpec{
			Privileged:               true,
			AllowPrivilegeEscalation: &allowPrivilegeEscalation,
			AllowedCapabilities:      []v1.Capability{"*"},
			Volumes:                  []policyv1beta1.FSType{policyv1beta1.All},
			HostNetwork:              true,
			HostPorts:                []policyv1beta1.HostPortRange{{Min: 0, Max: 65535}},
			HostIPC:                  true,
			HostPID:                  true,
			RunAsUser: policyv1beta1.RunAsUserStrategyOptions{
				Rule: policyv1beta1.RunAsUserStrategyRunAsAny,
			},
			SELinux: policyv1beta1.SELinuxStrategyOptions{
				Rule: policyv1beta1.SELinuxStrategyRunAsAny,
			},
			SupplementalGroups: policyv1beta1.SupplementalGroupsStrategyOptions{
				Rule: policyv1beta1.SupplementalGroupsStrategyRunAsAny,
			},
			FSGroup: policyv1beta1.FSGroupStrategyOptions{
				Rule: policyv1beta1.FSGroupStrategyRunAsAny,
			},
			ReadOnlyRootFilesystem: false,
			AllowedUnsafeSysctls:   []string{"*"},
		},
	}
}

// IsPodSecurityPolicyEnabled returns true if PodSecurityPolicy is enabled. Otherwise false.
func IsPodSecurityPolicyEnabled(kubeClient clientset.Interface) bool {
	isPSPEnabledOnce.Do(func() {
		psps, err := kubeClient.PolicyV1beta1().PodSecurityPolicies().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			Logf("Error listing PodSecurityPolicies; assuming PodSecurityPolicy is disabled: %v", err)
			return
		}
		if psps == nil || len(psps.Items) == 0 {
			Logf("No PodSecurityPolicies found; assuming PodSecurityPolicy is disabled.")
			return
		}
		Logf("Found PodSecurityPolicies; testing pod creation to see if PodSecurityPolicy is enabled")
		testPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "psp-test-pod-"},
			Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "test", Image: imageutils.GetPauseImageName()}}},
		}
		dryRunPod, err := kubeClient.CoreV1().Pods("kube-system").Create(context.TODO(), testPod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		if err != nil {
			if strings.Contains(err.Error(), "PodSecurityPolicy") {
				Logf("PodSecurityPolicy error creating dryrun pod; assuming PodSecurityPolicy is enabled: %v", err)
				isPSPEnabled = true
			} else {
				Logf("Error creating dryrun pod; assuming PodSecurityPolicy is disabled: %v", err)
			}
			return
		}
		pspAnnotation, pspAnnotationExists := dryRunPod.Annotations["kubernetes.io/psp"]
		if !pspAnnotationExists {
			Logf("No PSP annotation exists on dry run pod; assuming PodSecurityPolicy is disabled")
			return
		}
		Logf("PSP annotation exists on dry run pod: %q; assuming PodSecurityPolicy is enabled", pspAnnotation)
		isPSPEnabled = true
	})
	return isPSPEnabled
}

var (
	privilegedPSPOnce sync.Once
)

// CreatePrivilegedPSPBinding creates the privileged PSP & role
func CreatePrivilegedPSPBinding(kubeClient clientset.Interface, namespace string) {
	if !IsPodSecurityPolicyEnabled(kubeClient) {
		return
	}
	// Create the privileged PSP & role
	privilegedPSPOnce.Do(func() {
		_, err := kubeClient.PolicyV1beta1().PodSecurityPolicies().Get(context.TODO(), podSecurityPolicyPrivileged, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			// Privileged PSP was already created.
			ExpectNoError(err, "Failed to get PodSecurityPolicy %s", podSecurityPolicyPrivileged)
			return
		}

		psp := privilegedPSP(podSecurityPolicyPrivileged)
		_, err = kubeClient.PolicyV1beta1().PodSecurityPolicies().Create(context.TODO(), psp, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			ExpectNoError(err, "Failed to create PSP %s", podSecurityPolicyPrivileged)
		}

		if e2eauth.IsRBACEnabled(kubeClient.RbacV1()) {
			// Create the Role to bind it to the namespace.
			_, err = kubeClient.RbacV1().ClusterRoles().Create(context.TODO(), &rbacv1.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: podSecurityPolicyPrivileged},
				Rules: []rbacv1.PolicyRule{{
					APIGroups:     []string{"extensions"},
					Resources:     []string{"podsecuritypolicies"},
					ResourceNames: []string{podSecurityPolicyPrivileged},
					Verbs:         []string{"use"},
				}},
			}, metav1.CreateOptions{})
			if !apierrors.IsAlreadyExists(err) {
				ExpectNoError(err, "Failed to create PSP role")
			}
		}
	})

	if e2eauth.IsRBACEnabled(kubeClient.RbacV1()) {
		ginkgo.By(fmt.Sprintf("Binding the %s PodSecurityPolicy to the default service account in %s",
			podSecurityPolicyPrivileged, namespace))
		err := e2eauth.BindClusterRoleInNamespace(kubeClient.RbacV1(),
			podSecurityPolicyPrivileged,
			namespace,
			rbacv1.Subject{
				Kind:      rbacv1.ServiceAccountKind,
				Namespace: namespace,
				Name:      "default",
			},
			rbacv1.Subject{
				Kind:     rbacv1.GroupKind,
				APIGroup: rbacv1.GroupName,
				Name:     "system:serviceaccounts:" + namespace,
			},
		)
		ExpectNoError(err)
		ExpectNoError(e2eauth.WaitForNamedAuthorizationUpdate(kubeClient.AuthorizationV1(),
			serviceaccount.MakeUsername(namespace, "default"), namespace, "use", podSecurityPolicyPrivileged,
			schema.GroupResource{Group: "extensions", Resource: "podsecuritypolicies"}, true))
	}
}
