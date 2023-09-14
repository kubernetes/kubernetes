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

	authv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	kubeProxyServiceAccountName     = "kube-proxy"
	kubeProxyConfigMap              = "kube-proxy"
	kubeProxyConfigMapKey           = "config.conf"
	kubeProxyConfigMapKeyKubeconfig = "kubeconfig.conf"
	kubeProxyClusterRoleName        = "system:node-proxier"
	kubeProxyClusterRoleBindingName = "kubeadm:node-proxier"
	kubeProxyRoleName               = "kube-proxy"
	kubeProxyRoleBindingName        = kubeProxyRoleName
	kubeProxyDaemonSetName          = "kube-proxy"
)

var (
	kubeProxyConfigMapResource = &authv1.ResourceAttributes{
		Namespace: kubeSystemNamespace,
		Name:      kubeProxyConfigMap,
		Resource:  "configmaps",
		Verb:      "get",
	}
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the proxy addon as expected
var _ = Describe("proxy addon", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("proxy")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.Context("kube-proxy ServiceAccount", func() {
		ginkgo.It("should exist", func(ctx context.Context) {
			ExpectServiceAccount(f.ClientSet, kubeSystemNamespace, kubeProxyServiceAccountName)
		})

		ginkgo.It("should be bound to the system:node-proxier cluster role", func(ctx context.Context) {
			ExpectClusterRoleBindingWithSubjectAndRole(f.ClientSet,
				kubeProxyClusterRoleBindingName,
				rbacv1.ServiceAccountKind, kubeProxyServiceAccountName,
				kubeProxyClusterRoleName,
			)
		})
	})

	ginkgo.Context("kube-proxy ConfigMap", func() {
		ginkgo.It("should exist and be properly configured", func(ctx context.Context) {
			cm := GetConfigMap(f.ClientSet, kubeSystemNamespace, kubeProxyConfigMap)

			gomega.Expect(cm.Data).To(gomega.HaveKey(kubeProxyConfigMapKey))
			gomega.Expect(cm.Data).To(gomega.HaveKey(kubeProxyConfigMapKeyKubeconfig))
		})

		ginkgo.It("should have related Role and RoleBinding", func(ctx context.Context) {
			ExpectRole(f.ClientSet, kubeSystemNamespace, kubeProxyRoleName)
			ExpectRoleBinding(f.ClientSet, kubeSystemNamespace, kubeProxyRoleBindingName)
		})

		ginkgo.It("should be accessible by bootstrap tokens", func(ctx context.Context) {
			ExpectSubjectHasAccessToResource(f.ClientSet,
				rbacv1.GroupKind, bootstrapTokensGroup,
				kubeProxyConfigMapResource,
			)
		})
	})

	ginkgo.Context("kube-proxy DaemonSet", func() {
		ginkgo.It("should exist and be properly configured", func(ctx context.Context) {
			ds := GetDaemonSet(f.ClientSet, kubeSystemNamespace, kubeProxyDaemonSetName)

			gomega.Expect(ds.Spec.Template.Spec.ServiceAccountName).To(gomega.Equal(kubeProxyServiceAccountName))
		})
	})
})
