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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	nodesGroup                                 = "system:nodes"
	nodesCertificateRotationClusterRoleName    = "system:certificates.k8s.io:certificatesigningrequests:selfnodeclient"
	nodesCertificateRotationClusterRoleBinding = "kubeadm:node-autoapprove-certificate-rotation"
	nodesCRISocketAnnotation                   = "kubeadm.alpha.kubernetes.io/cri-socket"
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the nodes and system:nodes group as expected
var _ = Describe("nodes", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("nodes")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should have CRI annotation", func() {
		nodes, err := f.ClientSet.CoreV1().Nodes().
			List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "error reading nodes")

		// Checks that the nodes have the CRI socket annotation
		// and that it is prefixed with a URL scheme
		for _, node := range nodes.Items {
			gomega.Expect(node.Annotations).To(gomega.HaveKey(nodesCRISocketAnnotation))
			gomega.Expect(node.Annotations[nodesCRISocketAnnotation]).To(gomega.HavePrefix("unix://"))
		}
	})

	ginkgo.It("should be allowed to rotate CSR", func() {
		// Nb. this is technically implemented a part of the bootstrap-token phase
		ExpectClusterRoleBindingWithSubjectAndRole(f.ClientSet,
			nodesCertificateRotationClusterRoleBinding,
			rbacv1.GroupKind, nodesGroup,
			nodesCertificateRotationClusterRoleName,
		)
	})
})
