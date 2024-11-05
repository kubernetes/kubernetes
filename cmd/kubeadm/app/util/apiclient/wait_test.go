/*
Copyright 2024 The Kubernetes Authors.

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

package apiclient

import (
	"context"
	"fmt"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	clientfake "k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// TestGetControlPlaneComponents tests that getControlPlaneComponents returns the correct control plane components and their URLs.
func TestGetControlPlaneComponents(t *testing.T) {
	testcases := []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected []controlPlaneComponent
	}{
		{
			name: "port and addresses from config",
			cfg: &kubeadmapi.ClusterConfiguration{
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "secure-port", Value: "1111"},
							{Name: "bind-address", Value: "0.0.0.0"},
						},
					},
				},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "2222"},
						{Name: "bind-address", Value: "0.0.0.0"},
					},
				},
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "3333"},
						{Name: "bind-address", Value: "0.0.0.0"},
					},
				},
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: fmt.Sprintf("https://0.0.0.0:1111/%s", endpointLivez)},
				{name: "kube-controller-manager", url: fmt.Sprintf("https://0.0.0.0:2222/%s", endpointHealthz)},
				{name: "kube-scheduler", url: fmt.Sprintf("https://0.0.0.0:3333/%s", endpointLivez)},
			},
		},
		{
			name: "default ports and addresses",
			cfg:  &kubeadmapi.ClusterConfiguration{},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: fmt.Sprintf("https://127.0.0.1:6443/%s", endpointLivez)},
				{name: "kube-controller-manager", url: fmt.Sprintf("https://127.0.0.1:10257/%s", endpointHealthz)},
				{name: "kube-scheduler", url: fmt.Sprintf("https://127.0.0.1:10259/%s", endpointLivez)},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// Call the function under test
			actual := getControlPlaneComponents(tc.cfg)
			// Compare the expected and actual results
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Fatalf("expected result: %+v, got: %+v", tc.expected, actual)
			}
		})
	}
}

// TestGetStaticPodSingleHash tests that getStaticPodSingleHash correctly retrieves the static pod hash from annotations.
func TestGetStaticPodSingleHash(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	component := "kube-apiserver"
	podName := fmt.Sprintf("%s-%s", component, nodeName)
	expectedHash := "abc123"

	// Create annotations with the expected hash
	annotations := map[string]string{
		"kubernetes.io/config.hash": expectedHash,
	}

	// Create a pod with the annotations
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   metav1.NamespaceSystem,
			Annotations: annotations,
		},
	}

	// Create the pod in the fake clientset
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Call getStaticPodSingleHash and verify the result
	hash, err := getStaticPodSingleHash(client, nodeName, component)
	if err != nil {
		t.Errorf("getStaticPodSingleHash returned error: %v", err)
	}

	if hash != expectedHash {
		t.Errorf("Expected hash %s, got %s", expectedHash, hash)
	}
}

// TestGetStaticPodSingleHash_PodNotFound tests that getStaticPodSingleHash returns an error when the pod is not found.
func TestGetStaticPodSingleHash_PodNotFound(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	component := "kube-apiserver"

	// Call getStaticPodSingleHash without creating the pod
	_, err := getStaticPodSingleHash(client, nodeName, component)
	if err == nil {
		t.Errorf("Expected error when pod not found, got nil")
	}
}

// TestWaitForStaticPodSingleHash tests that WaitForStaticPodSingleHash successfully waits for and retrieves the static pod hash.
func TestWaitForStaticPodSingleHash(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	component := "kube-apiserver"
	podName := fmt.Sprintf("%s-%s", component, nodeName)
	expectedHash := "abc123"

	// Create annotations with the expected hash
	annotations := map[string]string{
		"kubernetes.io/config.hash": expectedHash,
	}

	// Create a pod with the annotations
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   metav1.NamespaceSystem,
			Annotations: annotations,
		},
	}

	// Create the pod in the fake clientset
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Create a KubeWaiter instance
	waiter := &KubeWaiter{
		client:  client,
		timeout: 5 * time.Second,
		writer:  io.Discard,
	}

	// Call WaitForStaticPodSingleHash and verify the result
	hash, err := waiter.WaitForStaticPodSingleHash(nodeName, component)
	if err != nil {
		t.Errorf("WaitForStaticPodSingleHash returned error: %v", err)
	}

	if hash != expectedHash {
		t.Errorf("Expected hash %s, got %s", expectedHash, hash)
	}
}

// TestWaitForPodsWithLabel tests that WaitForPodsWithLabel correctly waits for all pods with a label to be running.
func TestWaitForPodsWithLabel(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	// Create a KubeWaiter instance
	waiter := &KubeWaiter{
		client:  client,
		timeout: 5 * time.Second,
		writer:  io.Discard,
	}

	kvLabel := "app=test"

	// Create pods with the label but different statuses
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"app": "test"},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPending,
		},
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod2",
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"app": "test"},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}

	// Create the pods in the fake clientset
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod1: %v", err)
	}

	_, err = client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod2: %v", err)
	}

	// Expect an error because not all pods are running
	err = waiter.WaitForPodsWithLabel(kvLabel)
	if err == nil {
		t.Errorf("Expected WaitForPodsWithLabel to return an error because not all pods are running")
	}

	// Update pod1 status to Running
	pod1.Status.Phase = v1.PodRunning
	_, err = client.CoreV1().Pods(metav1.NamespaceSystem).Update(context.Background(), pod1, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod1: %v", err)
	}

	// Now the function should succeed
	err = waiter.WaitForPodsWithLabel(kvLabel)
	if err != nil {
		t.Errorf("WaitForPodsWithLabel returned error: %v", err)
	}
}

// TestWaitForStaticPodHashChange tests that WaitForStaticPodHashChange detects a hash change after updating the pod's annotations.
func TestWaitForStaticPodHashChange(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	component := "kube-apiserver"
	previousHash := "oldhash"
	podName := fmt.Sprintf("%s-%s", component, nodeName)

	// Create annotations with the previous hash
	annotations := map[string]string{
		"kubernetes.io/config.hash": previousHash,
	}

	// Create a pod with the previous hash
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   metav1.NamespaceSystem,
			Annotations: annotations,
		},
	}

	// Create the pod in the fake clientset
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Create a KubeWaiter instance
	waiter := &KubeWaiter{
		client:  client,
		timeout: 2 * time.Second,
		writer:  io.Discard,
	}

	// Simulate the hash change after some time
	go func() {
		time.Sleep(1 * time.Second)
		pod.Annotations["kubernetes.io/config.hash"] = "newhash"
		_, _ = client.CoreV1().Pods(metav1.NamespaceSystem).Update(context.Background(), pod, metav1.UpdateOptions{})
	}()

	// Call WaitForStaticPodHashChange and verify the result
	err = waiter.WaitForStaticPodHashChange(nodeName, component, previousHash)
	if err != nil {
		t.Errorf("WaitForStaticPodHashChange returned error: %v", err)
	}
}

// TestWaitForStaticPodHashChange_Timeout tests that WaitForStaticPodHashChange times out if the hash does not change.
func TestWaitForStaticPodHashChange_Timeout(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	component := "kube-apiserver"
	previousHash := "oldhash"
	podName := fmt.Sprintf("%s-%s", component, nodeName)

	// Create annotations with the previous hash
	annotations := map[string]string{
		"kubernetes.io/config.hash": previousHash,
	}

	// Create a pod with the previous hash
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   metav1.NamespaceSystem,
			Annotations: annotations,
		},
	}

	// Create the pod in the fake clientset
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// Create a KubeWaiter instance with a short timeout
	waiter := &KubeWaiter{
		client:  client,
		timeout: 1 * time.Second,
		writer:  io.Discard,
	}

	// Call WaitForStaticPodHashChange and expect it to timeout
	err = waiter.WaitForStaticPodHashChange(nodeName, component, previousHash)
	if err == nil {
		t.Errorf("Expected WaitForStaticPodHashChange to timeout but got nil")
	}
}

// TestWaitForStaticPodControlPlaneHashes tests that WaitForStaticPodControlPlaneHashes retrieves hashes for all control plane components.
func TestWaitForStaticPodControlPlaneHashes(t *testing.T) {
	// Initialize a fake clientset
	client := clientfake.NewSimpleClientset()

	nodeName := "node1"
	components := []string{"kube-apiserver", "kube-controller-manager", "kube-scheduler"}
	expectedHashes := map[string]string{}

	// Create pods for each component with unique hashes
	for i, component := range components {
		hash := fmt.Sprintf("hash%d", i)
		expectedHashes[component] = hash

		podName := fmt.Sprintf("%s-%s", component, nodeName)
		annotations := map[string]string{
			"kubernetes.io/config.hash": hash,
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        podName,
				Namespace:   metav1.NamespaceSystem,
				Annotations: annotations,
			},
		}

		_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.Background(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod for component %s: %v", component, err)
		}
	}

	// Create a KubeWaiter instance
	waiter := &KubeWaiter{
		client:  client,
		timeout: 5 * time.Second,
		writer:  io.Discard,
	}

	// Call WaitForStaticPodControlPlaneHashes and verify the result
	hashes, err := waiter.WaitForStaticPodControlPlaneHashes(nodeName)
	if err != nil {
		t.Errorf("WaitForStaticPodControlPlaneHashes returned error: %v", err)
	}

	if !reflect.DeepEqual(expectedHashes, hashes) {
		t.Errorf("Expected hashes %v, got %v", expectedHashes, hashes)
	}
}
