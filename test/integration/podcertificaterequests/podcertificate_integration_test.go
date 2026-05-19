/*
Copyright 2025 The Kubernetes Authors.

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

package podcertificaterequests

import (
	"testing"

	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"time"

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/certificates/cleaner"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/hermeticpodcertificatesigner"
	"k8s.io/utils/clock"
)

func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
}

func TestCleanerController(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1beta1.SchemeGroupVersion),
		},
		framework.SharedEtcd(),
	)
	defer s.TearDownFn()

	client := clientset.NewForConfigOrDie(s.ClientConfig)
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(s.ClientConfig, "certificatesigningrequest-informers")), time.Second)

	// Register the cleaner controller with a short configured timeout.  Within
	// 15 seconds the PCR should be deleted.
	cleanerClient, err := serviceAccountClient(s.ClientConfig, "kube-system", "podcertificaterequestcleaner")
	if err != nil {
		t.Fatalf("Unexpected error creating client that impersonates kube-system/podcertificaterequestcleaner: %v", err)
	}
	c := cleaner.NewPCRCleanerController(
		cleanerClient,
		informers.Certificates().V1beta1().PodCertificateRequests(),
		clock.RealClock{},
		1*time.Second,
		1*time.Second,
	)
	go c.Run(ctx, 1)

	// Start the controller & informers
	informers.Start(ctx.Done())

	// Make a node
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Spec: corev1.NodeSpec{},
	}
	node, err = client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}

	// Make a serviceaccount
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "sa1",
		},
	}
	sa, err = client.CoreV1().ServiceAccounts("default").Create(ctx, sa, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating sa1: %v", err)
	}

	// Make a pod
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod1",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: sa.ObjectMeta.Name,
			NodeName:           node.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
				},
			},
		},
	}
	pod, err = client.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating pod1: %v", err)
	}

	// Create a clientset that impersonates node1
	node1Client, err := nodeClient(s.ClientConfig, "node1")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	// Have node1 create a PodCertificateRequest for pod1
	_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(pod.ObjectMeta.UID))
	pcr := &certsv1beta1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pcr1",
		},
		Spec: certsv1beta1.PodCertificateRequestSpec{
			SignerName:                "kubernetes.io/foo",
			PodName:                   pod.ObjectMeta.Name,
			PodUID:                    pod.ObjectMeta.UID,
			ServiceAccountName:        sa.ObjectMeta.Name,
			ServiceAccountUID:         sa.ObjectMeta.UID,
			NodeName:                  types.NodeName(node.ObjectMeta.Name),
			NodeUID:                   node.ObjectMeta.UID,
			PKIXPublicKey:             pubPKIX,
			ProofOfPossession:         proof,
			UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "pod1"},
		},
	}
	pcr, err = node1Client.CertificatesV1beta1().PodCertificateRequests(pcr.ObjectMeta.Namespace).Create(ctx, pcr, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating PodCertificateRequest: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := client.CertificatesV1beta1().PodCertificateRequests(pcr.ObjectMeta.Namespace).Get(ctx, pcr.ObjectMeta.Name, metav1.GetOptions{})
		if k8serrors.IsNotFound(err) {
			return true, nil
		} else if err != nil {
			return false, err
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Unexpected error after waiting for PodCertificateRequest to be deleted: %v", err)
	}

	// TODO(KEP-4317): For beta, check via audit logs that it was the cleaner
	// controller that issued the deletion.
}

func TestNodeRestriction(t *testing.T) {
	// Create a setup with two nodes, and a pod running on node1.  Node2 cannot
	// make a PodCertificateRequest that refers to the pod on node1.
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--enable-admission-plugins=NodeRestriction",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1beta1.SchemeGroupVersion),
		},
		framework.SharedEtcd(),
	)
	defer s.TearDownFn()

	client := clientset.NewForConfigOrDie(s.ClientConfig)

	// Make node1 and node2
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Spec: corev1.NodeSpec{},
	}
	node1, err := client.CoreV1().Nodes().Create(ctx, node1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node2",
		},
		Spec: corev1.NodeSpec{},
	}
	node2, err = client.CoreV1().Nodes().Create(ctx, node2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}

	// Create clientsets for nodes
	node1Client, err := nodeClient(s.ClientConfig, "node1")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	node2Client, err := nodeClient(s.ClientConfig, "node2")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	// Make a serviceaccount
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "sa1",
		},
	}
	sa, err = client.CoreV1().ServiceAccounts("default").Create(ctx, sa, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating sa1: %v", err)
	}

	// Make a pod
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod1",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: sa.ObjectMeta.Name,
			NodeName:           node1.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
				},
			},
		},
	}
	pod, err = client.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating pod1: %v", err)
	}

	t.Run("node1 can create PCR for pod on node1", func(t *testing.T) {
		// Have node2 create a PodCertificateRequest for pod1
		_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(pod.ObjectMeta.UID))
		pcr := &certsv1beta1.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      "pcr1",
			},
			Spec: certsv1beta1.PodCertificateRequestSpec{
				SignerName:                "kubernetes.io/foo",
				PodName:                   pod.ObjectMeta.Name,
				PodUID:                    pod.ObjectMeta.UID,
				ServiceAccountName:        sa.ObjectMeta.Name,
				ServiceAccountUID:         sa.ObjectMeta.UID,
				NodeName:                  types.NodeName(node1.ObjectMeta.Name),
				NodeUID:                   node1.ObjectMeta.UID,
				PKIXPublicKey:             pubPKIX,
				ProofOfPossession:         proof,
				UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "pod1"},
			},
		}

		// Informer lag inside kube-apiserver could cause us to get transient
		// errors.
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
			_, err = node1Client.CertificatesV1beta1().PodCertificateRequests("default").Create(ctx, pcr, metav1.CreateOptions{})
			if err != nil {
				return false, err
			}
			return true, nil
		})
		if err != nil {
			t.Fatalf("PCR creation unexpectedly failed: %v", err)
		}
	})

	t.Run("node2 cannot create PCR for pod on node1", func(t *testing.T) {
		// Have node2 create a PodCertificateRequest for pod1
		_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(pod.ObjectMeta.UID))
		pcr := &certsv1beta1.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      "pcr1",
			},
			Spec: certsv1beta1.PodCertificateRequestSpec{
				SignerName:                "kubernetes.io/foo",
				PodName:                   pod.ObjectMeta.Name,
				PodUID:                    pod.ObjectMeta.UID,
				ServiceAccountName:        sa.ObjectMeta.Name,
				ServiceAccountUID:         sa.ObjectMeta.UID,
				NodeName:                  types.NodeName(node1.ObjectMeta.Name),
				NodeUID:                   node1.ObjectMeta.UID,
				PKIXPublicKey:             pubPKIX,
				ProofOfPossession:         proof,
				UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "pod1"},
			},
		}

		// Informer lag inside kube-apiserver could cause us to get a
		// non-Forbidden error from the noderestriction admission plugin.  This
		// should be transient, so wait for some time to see if we reach our
		// durable error condition.
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
			_, err = node2Client.CertificatesV1beta1().PodCertificateRequests("default").Create(ctx, pcr, metav1.CreateOptions{})
			if err == nil || k8serrors.IsForbidden(err) {
				return true, err
			}
			return false, err
		})
		if err == nil {
			t.Fatalf("PCR creation unexpectedly succeeded")
		} else if !k8serrors.IsForbidden(err) {
			t.Fatalf("PCR creation failed with unexpected error code (wanted Forbidden): %v", err)
		}
	})

	t.Run("node2 cannot create PCR for pod that doesn't exist", func(t *testing.T) {
		// Have node2 create a PodCertificateRequest for pod1
		_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(pod.ObjectMeta.UID))
		pcr := &certsv1beta1.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      "pcr1",
			},
			Spec: certsv1beta1.PodCertificateRequestSpec{
				SignerName:                "kubernetes.io/foo",
				PodName:                   "dnepod",
				PodUID:                    "dnepoduid",
				ServiceAccountName:        sa.ObjectMeta.Name,
				ServiceAccountUID:         sa.ObjectMeta.UID,
				NodeName:                  types.NodeName(node2.ObjectMeta.Name),
				NodeUID:                   node2.ObjectMeta.UID,
				PKIXPublicKey:             pubPKIX,
				ProofOfPossession:         proof,
				UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "pod1"},
			},
		}

		// The noderestriction admission plugin will *not* return Forbidden,
		// since this situation could always be caused by informer lag.  Just
		// hold here for 15 seconds and assume if we're still getting an error,
		// then it can't be due to informer lag.
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
			_, err = node2Client.CertificatesV1beta1().PodCertificateRequests("default").Create(ctx, pcr, metav1.CreateOptions{})
			if err == nil {
				return true, err
			}
			return false, nil
		})
		if err == nil { // EQUALS nil
			t.Fatalf("PCR creation unexpectedly succeeded")
		}

	})
}

func TestNodeAuthorization(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--enable-admission-plugins=NodeRestriction",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1beta1.SchemeGroupVersion),
		},
		framework.SharedEtcd(),
	)
	defer s.TearDownFn()

	client := clientset.NewForConfigOrDie(s.ClientConfig)

	// Make node1 and node2
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Spec: corev1.NodeSpec{},
	}
	node1, err := client.CoreV1().Nodes().Create(ctx, node1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node2",
		},
		Spec: corev1.NodeSpec{},
	}
	_, err = client.CoreV1().Nodes().Create(ctx, node2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}

	// Make a serviceaccount
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "sa1",
		},
	}
	sa, err = client.CoreV1().ServiceAccounts("default").Create(ctx, sa, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating sa1: %v", err)
	}

	// Make a pod
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod1",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: sa.ObjectMeta.Name,
			NodeName:           node1.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
				},
			},
		},
	}
	pod, err = client.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating pod1: %v", err)
	}

	// Create a clientsets that impersonate the nodes
	node1Client, err := nodeClient(s.ClientConfig, "node1")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	node2Client, err := nodeClient(s.ClientConfig, "node2")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	// Have node1 create a PodCertificateRequest for pod1
	_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(pod.ObjectMeta.UID))
	pcr := &certsv1beta1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pcr1",
		},
		Spec: certsv1beta1.PodCertificateRequestSpec{
			SignerName:                "kubernetes.io/foo",
			PodName:                   pod.ObjectMeta.Name,
			PodUID:                    pod.ObjectMeta.UID,
			ServiceAccountName:        sa.ObjectMeta.Name,
			ServiceAccountUID:         sa.ObjectMeta.UID,
			NodeName:                  types.NodeName(node1.ObjectMeta.Name),
			NodeUID:                   node1.ObjectMeta.UID,
			PKIXPublicKey:             pubPKIX,
			ProofOfPossession:         proof,
			UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "pod1"},
		},
	}

	// Creating the PCR could fail if there is informer lag in the
	// noderestriction logic.  Poll until it succeeds.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err = node1Client.CertificatesV1beta1().PodCertificateRequests("default").Create(ctx, pcr, metav1.CreateOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Unexpected error creating PodCertificateRequest: %v", err)
	}

	t.Run("node1 can directly get pcr1", func(t *testing.T) {
		_, err := node1Client.CertificatesV1beta1().PodCertificateRequests("default").Get(ctx, "pcr1", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error listing PodCertificateRequests as node1: %v", err)
		}
	})

	t.Run("node1 can see pcr1 when listing", func(t *testing.T) {
		pcrList, err := node1Client.CertificatesV1beta1().PodCertificateRequests("default").List(ctx, metav1.ListOptions{
			FieldSelector: "spec.nodeName=node1",
		})
		if err != nil {
			t.Fatalf("Unexpected error listing PodCertificateRequests as node1: %v", err)
		}

		if len(pcrList.Items) != 1 {
			t.Fatalf("Unexpected list length returned when node1 lists PodCertificateRequests; got %d, want 1", len(pcrList.Items))
		}

		if pcrList.Items[0].ObjectMeta.Name != "pcr1" {
			t.Fatalf("Unexpected list contents returned when node1 lists PodCertificateRequests; got %q want %q", pcrList.Items[0].ObjectMeta.Name, "pcr1")
		}
	})

	t.Run("node2 cannot list with field selector for node1", func(t *testing.T) {
		_, err := node2Client.CertificatesV1beta1().PodCertificateRequests("default").List(ctx, metav1.ListOptions{
			FieldSelector: "spec.nodeName=node1",
		})
		if err == nil {
			t.Fatalf("Listing PodCertificateRequests unexpectedly succeeded")
		} else if !k8serrors.IsForbidden(err) {
			t.Fatalf("Listing PodCertificateRequests failed with unexpected error (want Forbidden): %v", err)
		}
	})

	t.Run("node2 cannot directly get pcr1", func(t *testing.T) {
		_, err := node2Client.CertificatesV1beta1().PodCertificateRequests("default").Get(ctx, "pcr1", metav1.GetOptions{})
		if err == nil {
			t.Fatalf("Getting pcr1 unexpectedly succeeded")
		} else if !k8serrors.IsForbidden(err) {
			t.Fatalf("Getting pcr1 failed with unexpected error (want Forbidden): %v", err)
		}
	})

	t.Run("node2 cannot see pcr1 when listing", func(t *testing.T) {
		pcrList, err := node2Client.CertificatesV1beta1().PodCertificateRequests("default").List(ctx, metav1.ListOptions{
			FieldSelector: "spec.nodeName=node2",
		})
		if err != nil {
			t.Fatalf("Unexpected error listing PodCertificateRequests as node2: %v", err)
		}

		if len(pcrList.Items) != 0 {
			t.Fatalf("Unexpected list length returned when node2 lists PodCertificateRequests; got %d, want 0", len(pcrList.Items))
		}
	})
}

func TestNodeAuthorizerNamespaceNameConfusion(t *testing.T) {
	// A targeted test case to make sure that the node authorizer isn't mixing
	// up namespaces and names.

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--enable-admission-plugins=NodeRestriction",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1beta1.SchemeGroupVersion),
		},
		framework.SharedEtcd(),
	)
	defer s.TearDownFn()

	client := clientset.NewForConfigOrDie(s.ClientConfig)

	// Make node1 and node2
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Spec: corev1.NodeSpec{},
	}
	node1, err := client.CoreV1().Nodes().Create(ctx, node1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node2",
		},
		Spec: corev1.NodeSpec{},
	}
	node2, err = client.CoreV1().Nodes().Create(ctx, node2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}

	// Make namespaces "foo" and "bar"
	_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating namespace foo: %v", err)
	}
	_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bar",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating namespace nar: %v", err)
	}

	// Make a serviceaccount in each namespace
	saFoo := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "foo",
		},
	}
	saFoo, err = client.CoreV1().ServiceAccounts("foo").Create(ctx, saFoo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating serviceaccount foo: %v", err)
	}
	saBar := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "bar",
			Name:      "bar",
		},
	}
	saBar, err = client.CoreV1().ServiceAccounts("bar").Create(ctx, saBar, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating serviceaccount bar: %v", err)
	}

	// Make a pod named "foo" in namespace "bar", and vice-versa
	podBarFoo := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "bar",
			Name:      "foo",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: saBar.ObjectMeta.Name,
			NodeName:           node1.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
				},
			},
		},
	}
	podBarFoo, err = client.CoreV1().Pods("bar").Create(ctx, podBarFoo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating pod bar/foo: %v", err)
	}
	podFooBar := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: saFoo.ObjectMeta.Name,
			NodeName:           node2.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
				},
			},
		},
	}
	podFooBar, err = client.CoreV1().Pods("foo").Create(ctx, podFooBar, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating pod foo/bar: %v", err)
	}

	// Create a clientsets that impersonate the nodes
	node1Client, err := nodeClient(s.ClientConfig, "node1")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	node2Client, err := nodeClient(s.ClientConfig, "node2")
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	// Have node1 create a PodCertificateRequest for bar/foo
	_, _, pubPKIX, proof := mustMakeEd25519KeyAndProof(t, []byte(podBarFoo.ObjectMeta.UID))
	pcrBarFoo := &certsv1beta1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "bar",
			Name:      "foo",
		},
		Spec: certsv1beta1.PodCertificateRequestSpec{
			SignerName:                "kubernetes.io/foo",
			PodName:                   podBarFoo.ObjectMeta.Name,
			PodUID:                    podBarFoo.ObjectMeta.UID,
			ServiceAccountName:        saBar.ObjectMeta.Name,
			ServiceAccountUID:         saBar.ObjectMeta.UID,
			NodeName:                  types.NodeName(node1.ObjectMeta.Name),
			NodeUID:                   node1.ObjectMeta.UID,
			PKIXPublicKey:             pubPKIX,
			ProofOfPossession:         proof,
			UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "foo"},
		},
	}
	// Creating the PCR could fail if there is informer lag in the
	// noderestriction logic.  Poll until it succeeds.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err = node1Client.CertificatesV1beta1().PodCertificateRequests("bar").Create(ctx, pcrBarFoo, metav1.CreateOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Unexpected error creating pcr bar/foo: %v", err)
	}

	// Have node2 create a PodCertificateRequest for foo/bar
	_, _, pubPKIXFooBar, proofFooBar := mustMakeEd25519KeyAndProof(t, []byte(podFooBar.ObjectMeta.UID))
	pcrFooBar := &certsv1beta1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certsv1beta1.PodCertificateRequestSpec{
			SignerName:                "kubernetes.io/foo",
			PodName:                   podFooBar.ObjectMeta.Name,
			PodUID:                    podFooBar.ObjectMeta.UID,
			ServiceAccountName:        saFoo.ObjectMeta.Name,
			ServiceAccountUID:         saFoo.ObjectMeta.UID,
			NodeName:                  types.NodeName(node2.ObjectMeta.Name),
			NodeUID:                   node2.ObjectMeta.UID,
			PKIXPublicKey:             pubPKIXFooBar,
			ProofOfPossession:         proofFooBar,
			UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "bar"},
		},
	}
	// Creating the PCR could fail if there is informer lag in the
	// noderestriction logic.  Poll until it succeeds.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err = node2Client.CertificatesV1beta1().PodCertificateRequests("foo").Create(ctx, pcrFooBar, metav1.CreateOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Unexpected error creating pcr foo/bar: %v", err)
	}

	t.Run("node1 can directly get bar/foo", func(t *testing.T) {
		_, err := node1Client.CertificatesV1beta1().PodCertificateRequests("bar").Get(ctx, "foo", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error getting bar/foo as node1: %v", err)
		}
	})

	t.Run("node2 can directly get foo/bar", func(t *testing.T) {
		_, err := node2Client.CertificatesV1beta1().PodCertificateRequests("foo").Get(ctx, "bar", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error getting foo/bar as node2: %v", err)
		}
	})

	// Delete bar/foo
	err = client.CertificatesV1beta1().PodCertificateRequests("bar").Delete(ctx, "foo", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Unexpected error deleting pcr bar/foo: %v", err)
	}

	t.Run("node2 can still directly get foo/bar after bar/foo was deleted", func(t *testing.T) {
		_, err := node2Client.CertificatesV1beta1().PodCertificateRequests("foo").Get(ctx, "bar", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error getting foo/bar as node2: %v", err)
		}
	})
}

func serviceAccountClient(cfg *restclient.Config, ns, sa string) (*clientset.Clientset, error) {
	newCfg := restclient.CopyConfig(cfg)
	newCfg.Impersonate.UserName = fmt.Sprintf("system:serviceaccount:%s:%s", ns, sa)
	newCfg.Impersonate.Groups = []string{"system:authenticated", "system:serviceaccounts"}
	return clientset.NewForConfig(newCfg)
}

func nodeClient(cfg *restclient.Config, node string) (*clientset.Clientset, error) {
	newCfg := restclient.CopyConfig(cfg)
	newCfg.Impersonate.UserName = fmt.Sprintf("system:node:%s", node)
	newCfg.Impersonate.Groups = []string{"system:authenticated", "system:nodes"}
	return clientset.NewForConfig(newCfg)
}

func mustMakeEd25519KeyAndProof(t *testing.T, toBeSigned []byte) (ed25519.PrivateKey, ed25519.PublicKey, []byte, []byte) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating ed25519 key: %v", err)
	}
	pubPKIX, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		t.Fatalf("Error while marshaling PKIX public key: %v", err)
	}
	sig := ed25519.Sign(priv, toBeSigned)
	return priv, pub, pubPKIX, sig
}
