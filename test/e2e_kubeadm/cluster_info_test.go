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
	authv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	clusterInfoConfigMapName   = "cluster-info"
	clusterInfoRoleName        = "kubeadm:bootstrap-signer-clusterinfo"
	clusterInfoRoleBindingName = clusterInfoRoleName
)

var (
	clusterInfoConfigMapResource = &authv1.ResourceAttributes{
		Namespace: kubePublicNamespace,
		Name:      clusterInfoConfigMapName,
		Resource:  "configmaps",
		Verb:      "get",
	}
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the cluster-info ConfigMap, that it is properly configured
// and that all the related RBAC rules are in place
var _ = Describe("cluster-info ConfigMap", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("cluster-info")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should exist and be properly configured", func() {
		// Nb. this is technically implemented a part of the bootstrap-token phase
		cm := GetConfigMap(f.ClientSet, kubePublicNamespace, clusterInfoConfigMapName)

		gomega.Expect(cm.Data).To(gomega.HaveKey(gomega.HavePrefix(bootstrapapi.JWSSignatureKeyPrefix)))
		gomega.Expect(cm.Data).To(gomega.HaveKey(bootstrapapi.KubeConfigKey))

		//TODO: What else? server?
	})

	ginkgo.It("should have related Role and RoleBinding", func() {
		// Nb. this is technically implemented a part of the bootstrap-token phase
		ExpectRole(f.ClientSet, kubePublicNamespace, clusterInfoRoleName)
		ExpectRoleBinding(f.ClientSet, kubePublicNamespace, clusterInfoRoleBindingName)
	})

	ginkgo.It("should be accessible for anonymous", func() {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.UserKind, anonymousUser,
			clusterInfoConfigMapResource,
		)
	})
})
