/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	kubeletConfigConfigMapKey = "kubelet"
)

var (
	kubeletConfigConfigMapName   string
	kubeletConfigRoleName        string
	kubeletConfigRoleBindingName string

	// TODO: remove these versioned strings and related logic once the UnversionedKubeletConfigMap
	// feature gate goes GA:
	// https://github.com/kubernetes/kubeadm/issues/1582
	kubeletConfigConfigMapNameVersioned   string
	kubeletConfigRoleNameVersioned        string
	kubeletConfigRoleBindingNameVersioned string
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the kubelet-config ConfigMap, that it is properly configured
// and that all the related RBAC rules are in place
var _ = Describe("kubelet-config ConfigMap", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("kubelet-config")

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	// kubelet-config map is named using the kubernetesVersion as a suffix, and so
	// it is necessary to get it from the kubeadm-config ConfigMap before testing
	ginkgo.BeforeEach(func() {
		// if the kubelet-config map name is already known exit
		if kubeletConfigConfigMapName != "" {
			return
		}

		// gets the ClusterConfiguration from the kubeadm kubeadm-config ConfigMap as a untyped map
		m := getClusterConfiguration(f.ClientSet)

		// Extract the kubernetesVersion
		// TODO: remove this after the UnversionedKubeletConfigMap feature gate goes GA:
		// https://github.com/kubernetes/kubeadm/issues/1582
		// At that point parsing the k8s version will no longer be needed in this test.
		gomega.Expect(m).To(gomega.HaveKey("kubernetesVersion"))
		k8sVersionString := m["kubernetesVersion"].(string)
		k8sVersion, err := version.ParseSemantic(k8sVersionString)
		if err != nil {
			framework.Failf("error reading kubernetesVersion from %s ConfigMap: %v", kubeadmConfigName, err)
		}

		kubeletConfigConfigMapName = "kubelet-config"
		kubeletConfigRoleName = "kubeadm:kubelet-config"
		kubeletConfigRoleBindingName = kubeletConfigRoleName

		kubeletConfigConfigMapNameVersioned = fmt.Sprintf("kubelet-config-%d.%d", k8sVersion.Major(), k8sVersion.Minor())
		kubeletConfigRoleNameVersioned = fmt.Sprintf("kubeadm:kubelet-config-%d.%d", k8sVersion.Major(), k8sVersion.Minor())
		kubeletConfigRoleBindingNameVersioned = kubeletConfigRoleNameVersioned
	})

	ginkgo.It("should exist and be properly configured", func() {
		// TODO: switch to GetConfigMap once UnversionedKubeletConfigMap feature gate goes GA:
		// https://github.com/kubernetes/kubeadm/issues/1582
		cm, err := f.ClientSet.CoreV1().
			ConfigMaps(kubeSystemNamespace).
			Get(context.TODO(), kubeletConfigConfigMapName, metav1.GetOptions{})
		if err != nil {
			cm, err = f.ClientSet.CoreV1().
				ConfigMaps(kubeSystemNamespace).
				Get(context.TODO(), kubeletConfigConfigMapNameVersioned, metav1.GetOptions{})
			framework.ExpectNoError(err, "error getting ConfigMap %q or %q from namespace %q",
				kubeletConfigConfigMapName, kubeletConfigConfigMapNameVersioned, kubeSystemNamespace)
		}
		gomega.Expect(cm.Data).To(gomega.HaveKey(kubeletConfigConfigMapKey))
	})

	ginkgo.It("should have related Role and RoleBinding", func() {
		// TODO: switch to ExpectRole(Binding) once UnversionedKubeletConfigMap feature gate goes GA:
		// https://github.com/kubernetes/kubeadm/issues/1582
		_, err := f.ClientSet.RbacV1().
			Roles(kubeSystemNamespace).
			Get(context.TODO(), kubeletConfigRoleName, metav1.GetOptions{})
		if err != nil {
			_, err = f.ClientSet.RbacV1().
				Roles(kubeSystemNamespace).
				Get(context.TODO(), kubeletConfigRoleNameVersioned, metav1.GetOptions{})
			framework.ExpectNoError(err, "error getting Role %q or %q from namespace %q",
				kubeletConfigRoleName, kubeletConfigRoleNameVersioned, kubeSystemNamespace)
		}
		_, err = f.ClientSet.RbacV1().
			Roles(kubeSystemNamespace).
			Get(context.TODO(), kubeletConfigRoleBindingName, metav1.GetOptions{})
		if err != nil {
			_, err = f.ClientSet.RbacV1().
				Roles(kubeSystemNamespace).
				Get(context.TODO(), kubeletConfigRoleBindingNameVersioned, metav1.GetOptions{})
			framework.ExpectNoError(err, "error getting RoleBinding %q or %q from namespace %q",
				kubeletConfigRoleBindingName, kubeletConfigRoleBindingNameVersioned, kubeSystemNamespace)
		}
	})

	ginkgo.It("should be accessible for bootstrap tokens", func() {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, bootstrapTokensGroup,
			kubeadmConfigConfigMapResource,
		)
	})

	ginkgo.It("should be accessible for nodes", func() {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, nodesGroup,
			kubeadmConfigConfigMapResource,
		)
	})
})
