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

	authv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	kubeadmCertsSecretName = "kubeadm-certs"
)

var (
	kubeadmCertsRoleName        = fmt.Sprintf("kubeadm:%s", kubeadmCertsSecretName)
	kubeadmCertsRoleBindingName = kubeadmCertsRoleName

	kubeadmCertsSecretResource = &authv1.ResourceAttributes{
		Namespace: kubeSystemNamespace,
		Name:      kubeadmCertsSecretName,
		Resource:  "secrets",
		Verb:      "get",
	}
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the kubeadm-certs Secret, that it is properly configured
// and that all the related RBAC rules are in place

// Important! please note that kubeadm-certs is not created by default (still alpha)
// in case you want to skip this test use SKIP=copy-certs
var _ = Describe("kubeadm-certs [copy-certs]", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("kubeadm-certs")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should exist and be properly configured", func(ctx context.Context) {
		s := GetSecret(f.ClientSet, kubeSystemNamespace, kubeadmCertsSecretName)

		// Checks the kubeadm-certs is ownen by a time lived token
		gomega.Expect(s.OwnerReferences).To(gomega.HaveLen(1), "%s should have one owner reference", kubeadmCertsSecretName)
		ownRef := s.OwnerReferences[0]
		gomega.Expect(ownRef.Kind).To(gomega.Equal("Secret"), "%s should be owned by a secret", kubeadmCertsSecretName)
		gomega.Expect(*ownRef.BlockOwnerDeletion).To(gomega.BeTrueBecause("%s should be deleted on owner deletion", kubeadmCertsSecretName))

		o := GetSecret(f.ClientSet, kubeSystemNamespace, ownRef.Name)
		gomega.Expect(o.Type).To(gomega.Equal(corev1.SecretTypeBootstrapToken), "%s should have an owner reference that refers to a bootstrap-token", kubeadmCertsSecretName)
		gomega.Expect(o.Data).To(gomega.HaveKey("expiration"), "%s should have an owner reference with an expiration", kubeadmCertsSecretName)

		// gets the ClusterConfiguration from the kubeadm kubeadm-config ConfigMap as a untyped map
		m := getClusterConfiguration(f.ClientSet)

		// Extract the etcd Type
		etcdType := "local"
		if _, ok := m["etcd"]; ok {
			d := m["etcd"].(map[interface{}]interface{})
			if _, ok := d["external"]; ok {
				etcdType = "external"
			}
		}

		// check if all the expected key exists
		gomega.Expect(s.Data).To(gomega.HaveKey("ca.crt"))
		gomega.Expect(s.Data).To(gomega.HaveKey("ca.key"))
		gomega.Expect(s.Data).To(gomega.HaveKey("front-proxy-ca.crt"))
		gomega.Expect(s.Data).To(gomega.HaveKey("front-proxy-ca.key"))
		gomega.Expect(s.Data).To(gomega.HaveKey("sa.pub"))
		gomega.Expect(s.Data).To(gomega.HaveKey("sa.key"))

		if etcdType == "local" {
			gomega.Expect(s.Data).To(gomega.HaveKey("etcd-ca.crt"))
			gomega.Expect(s.Data).To(gomega.HaveKey("etcd-ca.key"))
		} else {
			gomega.Expect(s.Data).To(gomega.HaveKey("external-etcd-ca.crt"))
			gomega.Expect(s.Data).To(gomega.HaveKey("external-etcd.crt"))
			gomega.Expect(s.Data).To(gomega.HaveKey("external-etcd.key"))
		}
	})

	ginkgo.It("should have related Role and RoleBinding", func(ctx context.Context) {
		ExpectRole(f.ClientSet, kubeSystemNamespace, kubeadmCertsRoleName)
		ExpectRoleBinding(f.ClientSet, kubeSystemNamespace, kubeadmCertsRoleBindingName)
	})

	ginkgo.It("should be accessible for bootstrap tokens", func(ctx context.Context) {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, bootstrapTokensGroup,
			kubeadmCertsSecretResource,
		)
	})
})
