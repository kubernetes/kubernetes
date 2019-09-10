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
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	bootstrapTokensGroup                             = "system:bootstrappers:kubeadm:default-node-token"
	bootstrapTokensAllowPostCSRClusterRoleBinding    = "kubeadm:kubelet-bootstrap"
	bootstrapTokensAllowPostCSRClusterRoleName       = "system:node-bootstrapper"
	bootstrapTokensCSRAutoApprovalClusterRoleBinding = "kubeadm:node-autoapprove-bootstrap"
	bootstrapTokensCSRAutoApprovalClusterRoleName    = "system:certificates.k8s.io:certificatesigningrequests:nodeclient"
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the bootstrap token, the system:bootstrappers:kubeadm:default-node-token group
// and that all the related RBAC rules are in place
var _ = KubeadmDescribe("bootstrap token", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("bootstrap token")

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should exist and be properly configured", func() {
		secrets, err := f.ClientSet.CoreV1().
			Secrets(kubeSystemNamespace).
			List(metav1.ListOptions{})
		framework.ExpectNoError(err, "error reading Secrets")

		tokenNum := 0
		for _, s := range secrets.Items {
			if s.Type == corev1.SecretTypeBootstrapToken {
				//TODO: might be we want to further check tokens  (auth-extra-groups, usage etc)
				tokenNum++
			}
		}
		gomega.Expect(tokenNum).Should(gomega.BeNumerically(">", 0), "At least one bootstrap token should exist")
	})

	ginkgo.It("should be allowed to post CSR for kubelet certificates on joining nodes", func() {
		ExpectClusterRoleBindingWithSubjectAndRole(f.ClientSet,
			bootstrapTokensAllowPostCSRClusterRoleBinding,
			rbacv1.GroupKind, bootstrapTokensGroup,
			bootstrapTokensAllowPostCSRClusterRoleName,
		)
		//TODO: check if possible to verify "allowed to post CSR" using subject asses review as well
	})

	ginkgo.It("should be allowed to auto approve CSR for kubelet certificates on joining nodes", func() {
		ExpectClusterRoleBindingWithSubjectAndRole(f.ClientSet,
			bootstrapTokensCSRAutoApprovalClusterRoleBinding,
			rbacv1.GroupKind, bootstrapTokensGroup,
			bootstrapTokensCSRAutoApprovalClusterRoleName,
		)
		//TODO: check if possible to verify "allowed to auto approve CSR" using subject asses review as well
	})
})
