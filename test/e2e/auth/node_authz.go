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
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	nodesGroup     = "system:nodes"
	nodeNamePrefix = "system:node:"
)

var _ = SIGDescribe(feature.NodeAuthorizer, func() {

	f := framework.NewDefaultFramework("node-authz")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	// client that will impersonate a node
	var c clientset.Interface
	var ns string
	var asUser string
	var nodeName string
	ginkgo.BeforeEach(func(ctx context.Context) {
		ns = f.Namespace.Name

		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list nodes in namespace: %s", ns)
		gomega.Expect(nodeList.Items).NotTo(gomega.BeEmpty())
		nodeName = nodeList.Items[0].Name
		asUser = nodeNamePrefix + nodeName
		ginkgo.By("Creating a kubernetes client that impersonates a node")
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "failed to load kubernetes client config")
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: asUser,
			Groups:   []string{nodesGroup},
		}
		c, err = clientset.NewForConfig(config)
		framework.ExpectNoError(err, "failed to create Clientset for the given config: %+v", *config)

	})
	ginkgo.It("Getting a non-existent secret should exit with the Forbidden error, not a NotFound error", func(ctx context.Context) {
		_, err := c.CoreV1().Secrets(ns).Get(ctx, "foo", metav1.GetOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})

	ginkgo.It("Getting an existing secret should exit with the Forbidden error", func(ctx context.Context) {
		ginkgo.By("Create a secret for testing")
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
				Name:      "node-auth-secret",
			},
			StringData: map[string]string{},
		}
		_, err := f.ClientSet.CoreV1().Secrets(ns).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create secret (%s:%s) %+v", ns, secret.Name, *secret)
		_, err = c.CoreV1().Secrets(ns).Get(ctx, secret.Name, metav1.GetOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})

	ginkgo.It("Getting a non-existent configmap should exit with the Forbidden error, not a NotFound error", func(ctx context.Context) {
		_, err := c.CoreV1().ConfigMaps(ns).Get(ctx, "foo", metav1.GetOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})

	ginkgo.It("Getting an existing configmap should exit with the Forbidden error", func(ctx context.Context) {
		ginkgo.By("Create a configmap for testing")
		configmap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
				Name:      "node-auth-configmap",
			},
			Data: map[string]string{
				"data": "content",
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(ns).Create(ctx, configmap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create configmap (%s:%s) %+v", ns, configmap.Name, *configmap)
		_, err = c.CoreV1().ConfigMaps(ns).Get(ctx, configmap.Name, metav1.GetOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})

	ginkgo.It("Getting a secret for a workload the node has access to should succeed", func(ctx context.Context) {
		ginkgo.By("Create a secret for testing")
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
				Name:      "node-auth-secret",
			},
			Data: map[string][]byte{
				"data": []byte("keep it secret"),
			},
		}
		_, err := f.ClientSet.CoreV1().Secrets(ns).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create secret (%s:%s)", ns, secret.Name)

		ginkgo.By("Node should not get the secret")
		_, err = c.CoreV1().Secrets(ns).Get(ctx, secret.Name, metav1.GetOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}

		ginkgo.By("Create a pod that use the secret")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pause",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: imageutils.GetPauseImageName(),
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

		_, err = f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod (%s:%s)", ns, pod.Name)

		ginkgo.By("The node should able to access the secret")
		itv := framework.Poll
		dur := 1 * time.Minute
		err = wait.Poll(itv, dur, func() (bool, error) {
			_, err = c.CoreV1().Secrets(ns).Get(ctx, secret.Name, metav1.GetOptions{})
			if err != nil {
				framework.Logf("Failed to get secret %v, err: %v", secret.Name, err)
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err, "failed to get secret after trying every %v for %v (%s:%s)", itv, dur, ns, secret.Name)
	})

	ginkgo.It("A node shouldn't be able to create another node", func(ctx context.Context) {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			TypeMeta: metav1.TypeMeta{
				Kind:       "Node",
				APIVersion: "v1",
			},
		}
		ginkgo.By(fmt.Sprintf("Create node foo by user: %v", asUser))
		_, err := c.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})

		// NOTE: If the test fails and a new node IS created, we need to delete it. If we don't, we'd have
		// a zombie node in a NotReady state which will delay further tests since we're waiting for all
		// tests to be in the Ready state.
		ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.CoreV1().Nodes().Delete), node.Name, metav1.DeleteOptions{})

		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})

	ginkgo.It("A node shouldn't be able to delete another node", func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("Create node foo by user: %v", asUser))
		err := c.CoreV1().Nodes().Delete(ctx, "foo", metav1.DeleteOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("should be a forbidden error, got %#v", err)
		}
	})
})
