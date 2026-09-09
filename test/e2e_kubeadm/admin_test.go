/*
Copyright 2023 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/util/version"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

const (
	kubeadmClusterAdminsGroupAndCRB = "kubeadm:cluster-admins"
	clusterAdminClusterRole         = "cluster-admin"
)

// Define a container for all the tests aimed at verifying that administrators
// have "cluster-admin" level of access.
var _ = Describe("admin", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("admin")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("kubeadm:cluster-admins CRB must exist and be binding the cluster-admin ClusterRole"+
		" to the kubeadm:cluster-admins Group", func(ctx context.Context) {

		// Use the ClusterConfiguration.kubernetesVersion to decide whether to look for the CRB.
		// It was added in kubeadm v1.29-pre, but kubeadm supports a maximum skew of N-1
		// with the control plane, so this is fine as long as kubernetesVersion matches
		// the kubeadm version. Otherwise this test is skipped.
		m := getClusterConfiguration(f.ClientSet)
		const verKey = "kubernetesVersion"
		verInterface, exists := m[verKey]
		if !exists {
			framework.Failf("the kubeadm ConfigMap %s is missing a %s key in %s",
				kubeadmConfigName,
				verKey,
				kubeadmConfigClusterConfigurationConfigMapKey)
		}

		// Parse the version string
		verStr, ok := verInterface.(string)
		if !ok {
			framework.Failf("cannot cast %s to string", verKey)
		}
		ver, err := version.ParseSemantic(verStr)
		if err != nil {
			framework.Failf("could not parse the %s key: %v", verKey, err)
		}

		// If the version is older than the version when this feature was added, skip this test
		minVer := version.MustParseSemantic("v1.29.0-alpha.2.188+05076de57fc49f")
		if !ver.AtLeast(minVer) {
			e2eskipper.Skipf("Skipping because version %s is older than the minimum version %s",
				ver,
				minVer)
		}

		// Expect the CRB to exist
		ExpectClusterRoleBindingWithSubjectAndRole(f.ClientSet,
			kubeadmClusterAdminsGroupAndCRB,
			rbacv1.GroupKind, kubeadmClusterAdminsGroupAndCRB,
			clusterAdminClusterRole,
		)
	})
})
