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
	"fmt"
	"sync"

	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/onsi/ginkgo"
)

const (
	PodSecurityPolicyPrivileged     = "privileged"
	PodSecurityPolicyPrivilegedRole = "podsecuritypolicy:privileged"
)

var (
	isPSPEnabledOnce sync.Once
	isPSPEnabled     bool
)

func IsPodSecurityPolicyEnabled(f *Framework) bool {
	isPSPEnabledOnce.Do(func() {
		psp, err := f.ClientSet.ExtensionsV1beta1().PodSecurityPolicies().
			Get(PodSecurityPolicyPrivileged, metav1.GetOptions{})
		if err != nil {
			Logf("Error getting PodSecurityPolicy %s; assuming PodSecurityPolicy is disabled: %v",
				PodSecurityPolicyPrivileged, err)
			isPSPEnabled = false
		} else if psp == nil {
			Logf("PodSecurityPolicy %s was not found; assuming PodSecurityPolicy is disabled.",
				PodSecurityPolicyPrivileged)
			isPSPEnabled = false
		} else {
			Logf("Found PodSecurityPolicy %s; assuming PodSecurityPolicy is enabled.",
				PodSecurityPolicyPrivileged)
			isPSPEnabled = true
		}
	})
	return isPSPEnabled
}

func CreateDefaultPSPBinding(f *Framework, namespace string) {
	By(fmt.Sprintf("Binding the %s PodSecurityPolicy to the default service account in %s",
		PodSecurityPolicyPrivileged, namespace))
	BindClusterRoleInNamespace(f.ClientSet.RbacV1beta1(),
		PodSecurityPolicyPrivilegedRole,
		namespace,
		rbacv1beta1.Subject{
			Kind:      rbacv1beta1.ServiceAccountKind,
			Namespace: namespace,
			Name:      "default",
		})
}
