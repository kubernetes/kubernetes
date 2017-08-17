/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	NodesGroup     = "system:nodes"
	NodeNamePrefix = "system:node:"
)

var _ = SIGDescribe("[Feature:NodeAuthorizer]", func() {

	f := framework.NewDefaultFramework("node-authz")
	// client that will impersonate a node
	var c clientset.Interface
	var ns string
	var asUser string
	var defaultSaSecret string
	var nodeName string
	BeforeEach(func() {
		ns = f.Namespace.Name

		nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(len(nodeList.Items)).NotTo(Equal(0))
		nodeName = nodeList.Items[0].Name
		asUser = NodeNamePrefix + nodeName
		sa, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Get("default", metav1.GetOptions{})
		Expect(len(sa.Secrets)).NotTo(Equal(0))
		Expect(err).NotTo(HaveOccurred())
		defaultSaSecret = sa.Secrets[0].Name
		By("Creating a kubernetes client that impersonates a node")
		config, err := framework.LoadConfig()
		Expect(err).NotTo(HaveOccurred())
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: asUser,
			Groups:   []string{NodesGroup},
		}
		c, err = clientset.NewForConfig(config)
		Expect(err).NotTo(HaveOccurred())

	})
	It("Getting a non-existent secret should exit with the Forbidden error, not a NotFound error", func() {
		_, err := c.CoreV1().Secrets(ns).Get("foo", metav1.GetOptions{})
		Expect(apierrors.IsForbidden(err)).Should(Equal(true))
	})

	It("Getting an existent secret should exit with the Forbidden error", func() {
		_, err := c.CoreV1().Secrets(ns).Get(defaultSaSecret, metav1.GetOptions{})
		Expect(apierrors.IsForbidden(err)).Should(Equal(true))
	})

	It("Getting a secret for a workload the node has access to should succeed", func() {
		By("Create a secret for testing")
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
				Name:      "node-auth-secret",
			},
			Data: map[string][]byte{
				"data": []byte("keep it secret"),
			},
		}
		_, err := f.ClientSet.CoreV1().Secrets(ns).Create(secret)
		Expect(err).NotTo(HaveOccurred())

		By("Node should not get the secret")
		_, err = c.CoreV1().Secrets(ns).Get(secret.Name, metav1.GetOptions{})
		Expect(apierrors.IsForbidden(err)).Should(Equal(true))

		By("Create a pod that use the secret")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pause",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: framework.GetPauseImageName(f.ClientSet),
					},
				},
				NodeName: nodeName,
				Volumes: []v1.Volume{
					{
						Name: "node-auth-secret",
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: secret.Name,
							},
						},
					},
				},
			},
		}

		_, err = f.ClientSet.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("The node should able to access the secret")
		err = wait.Poll(framework.Poll, 1*time.Minute, func() (bool, error) {
			_, err = c.CoreV1().Secrets(ns).Get(secret.Name, metav1.GetOptions{})
			if err != nil {
				framework.Logf("Failed to get secret %v, err: %v", secret.Name, err)
				return false, nil
			}
			return true, nil
		})
		Expect(err).NotTo(HaveOccurred())
	})

	It("A node shouldn't be able to create an other node", func() {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			TypeMeta: metav1.TypeMeta{
				Kind:       "Node",
				APIVersion: "v1",
			},
		}
		By(fmt.Sprintf("Create node foo by user: %v", asUser))
		_, err := c.CoreV1().Nodes().Create(node)
		Expect(apierrors.IsForbidden(err)).Should(Equal(true))
	})

	It("A node shouldn't be able to delete an other node", func() {
		By(fmt.Sprintf("Create node foo by user: %v", asUser))
		err := c.CoreV1().Nodes().Delete("foo", &metav1.DeleteOptions{})
		Expect(apierrors.IsForbidden(err)).Should(Equal(true))
	})
})
