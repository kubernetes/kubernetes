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

package e2e_kubeadm

import (
	"fmt"

	authv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"

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

	kubeletConfigConfigMapResource = &authv1.ResourceAttributes{
		Namespace: kubeSystemNamespace,
		Name:      "",
		Resource:  "configmaps",
		Verb:      "get",
	}
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the kubelet-config ConfigMap, that it is properly configured
// and that all the related RBAC rules are in place
var _ = KubeadmDescribe("kubelet-config ConfigMap", func() {

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
		gomega.Expect(m).To(gomega.HaveKey("kubernetesVersion"))
		k8sVersionString := m["kubernetesVersion"].(string)
		k8sVersion, err := version.ParseSemantic(k8sVersionString)
		if err != nil {
			e2elog.Failf("error reading kubernetesVersion from %s ConfigMap: %v", kubeadmConfigName, err)
		}

		// Computes all the names derived from the kubernetesVersion
		kubeletConfigConfigMapName = fmt.Sprintf("kubelet-config-%d.%d", k8sVersion.Major(), k8sVersion.Minor())
		kubeletConfigRoleName = fmt.Sprintf("kubeadm:kubelet-config-%d.%d", k8sVersion.Major(), k8sVersion.Minor())
		kubeletConfigRoleBindingName = kubeletConfigRoleName
		kubeletConfigConfigMapResource.Name = kubeletConfigConfigMapName
	})

	ginkgo.It("should exist and be properly configured", func() {
		cm := GetConfigMap(f.ClientSet, kubeSystemNamespace, kubeletConfigConfigMapName)

		gomega.Expect(cm.Data).To(gomega.HaveKey(kubeletConfigConfigMapKey))
	})

	ginkgo.It("should have related Role and RoleBinding", func() {
		ExpectRole(f.ClientSet, kubeSystemNamespace, kubeletConfigRoleName)
		ExpectRoleBinding(f.ClientSet, kubeSystemNamespace, kubeletConfigRoleBindingName)
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
