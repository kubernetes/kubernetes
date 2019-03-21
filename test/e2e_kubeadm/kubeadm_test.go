/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	masterTaint                  = "node-role.kubernetes.io/master"
	kubeadmConfigNamespace       = "kube-system"
	kubeadmConfigName            = "kubeadm-config"
	clusterInfoNamespace         = "kube-public"
	clusterInfoName              = "cluster-info"
	bootstrapSignerRoleNamespace = "kube-system"
	bootstrapSignerRoleName      = "system:controller:bootstrap-signer"
)

var _ = framework.KubeDescribe("Kubeadm [Feature:Kubeadm]", func() {
	f := framework.NewDefaultFramework("kubeadm")

	Describe("kubeadm master", func() {
		It("should be labelled and tainted", func() {
			selector := labels.Set{masterTaint: ""}.AsSelector()
			master, err := f.ClientSet.CoreV1().Nodes().
				List(metav1.ListOptions{LabelSelector: selector.String()})
			framework.ExpectNoError(err, "couldn't find a master node")
			Expect(master.Items).NotTo(BeEmpty())
			for _, master := range master.Items {
				Expect(master.Spec.Taints).To(
					ContainElement(taint(masterTaint, corev1.TaintEffectNoSchedule)),
				)
			}
		})
	})

	Describe("kubeadm-config config map", func() {
		It("should exist", func() {
			_, err := f.ClientSet.CoreV1().
				ConfigMaps(kubeadmConfigNamespace).
				Get(kubeadmConfigName, metav1.GetOptions{})
			framework.ExpectNoError(err)
		})
	})

	Describe("cluster-info", func() {
		It("should have expected keys", func() {
			clientInfo, err := f.ClientSet.CoreV1().
				ConfigMaps(clusterInfoNamespace).
				Get(clusterInfoName, metav1.GetOptions{})
			framework.ExpectNoError(err, "couldn't find config map")

			Expect(clientInfo.Data).To(HaveKey(HavePrefix(bootstrapapi.JWSSignatureKeyPrefix)))
			Expect(clientInfo.Data).To(HaveKey(bootstrapapi.KubeConfigKey))
		})

		It("should be public", func() {
			cfg, err := framework.LoadConfig()
			framework.ExpectNoError(err, "couldn't get config")
			cfg = rest.AnonymousClientConfig(cfg)
			client, err := kubernetes.NewForConfig(cfg)
			framework.ExpectNoError(err, "couldn't create client")

			_, err = client.CoreV1().ConfigMaps(clusterInfoNamespace).
				Get(clusterInfoName, metav1.GetOptions{})
			framework.ExpectNoError(err, "couldn't anonymously access config")
		})
	})

	Describe("bootstrap signer RBAC role", func() {
		It("should exist", func() {
			_, err := f.ClientSet.RbacV1().
				Roles(bootstrapSignerRoleNamespace).
				Get(bootstrapSignerRoleName, metav1.GetOptions{})
			framework.ExpectNoError(err, "doesn't exist")
		})
	})

	Describe("kubeadm:kubelet-bootstrap cluster role binding", func() {
		It("should exist", func() {
			binding, err := f.ClientSet.RbacV1().
				ClusterRoleBindings().
				Get("kubeadm:kubelet-bootstrap", metav1.GetOptions{})
			framework.ExpectNoError(err, "couldn't get clusterrolebinding")
			Expect(binding.Subjects).To(
				ContainElement(subject(
					"system:bootstrappers:kubeadm:default-node-token",
					rbacv1.GroupKind,
				)),
			)
			Expect(binding.RoleRef.Name).To(Equal("system:node-bootstrapper"))
		})
	})

	Describe("autoapproval for new bootstrap token", func() {
		It("should create a clusterrolebinding", func() {
			binding, err := f.ClientSet.RbacV1().
				ClusterRoleBindings().
				Get("kubeadm:node-autoapprove-bootstrap", metav1.GetOptions{})
			framework.ExpectNoError(err, "couldn't get clusterrolebinding")
			Expect(binding.Subjects).To(
				ContainElement(subject(
					"system:bootstrappers:kubeadm:default-node-token",
					rbacv1.GroupKind,
				)),
			)
			Expect(binding.RoleRef.Name).To(
				Equal("system:certificates.k8s.io:certificatesigningrequests:nodeclient"),
			)
		})
	})
})
