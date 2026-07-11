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

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	kubeletConfigConfigMapKey = "kubelet"
)

var (
	kubeletConfigConfigMapName   string
	kubeletConfigRoleName        string
	kubeletConfigRoleBindingName string
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the kubelet-config ConfigMap, that it is properly configured
// and that all the related RBAC rules are in place
var _ = Describe("kubelet-config ConfigMap", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("kubelet-config")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

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

		kubeletConfigConfigMapName = "kubelet-config"
		kubeletConfigRoleName = "kubeadm:kubelet-config"
		kubeletConfigRoleBindingName = kubeletConfigRoleName
	})

	ginkgo.It("should exist and be properly configured", func(ctx context.Context) {
		cm := GetConfigMap(f.ClientSet, kubeSystemNamespace, kubeletConfigConfigMapName)
		gomega.Expect(cm.Data).To(gomega.HaveKey(kubeletConfigConfigMapKey))
	})

	ginkgo.It("should have related Role and RoleBinding", func(ctx context.Context) {
		ExpectRole(f.ClientSet, kubeSystemNamespace, kubeletConfigRoleName)
		ExpectRoleBinding(f.ClientSet, kubeSystemNamespace, kubeletConfigRoleBindingName)
	})

	ginkgo.It("should be accessible for bootstrap tokens", func(ctx context.Context) {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, bootstrapTokensGroup,
			kubeadmConfigConfigMapResource,
		)
	})

	ginkgo.It("should be accessible for nodes", func(ctx context.Context) {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, nodesGroup,
			kubeadmConfigConfigMapResource,
		)
	})
})
