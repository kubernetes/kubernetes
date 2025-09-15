/*
Copyright 2016 The Kubernetes Authors.

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

package tests

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestFakeClientSetFiltering(t *testing.T) {
	tc := clientsetfake.NewSimpleClientset(
		testPod("nsA", "pod-1"),
		testPod("nsB", "pod-2"),
		testSA("nsA", "sa-1"),
		testSA("nsA", "sa-2"),
		testSA("nsB", "sa-1"),
		testSA("nsB", "sa-2"),
		testSA("nsB", "sa-3"),
	)

	saList1, err := tc.CoreV1().ServiceAccounts("nsA").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("ServiceAccounts.List: %s", err)
	}
	if actual, expected := len(saList1.Items), 2; expected != actual {
		t.Fatalf("Expected %d records to match, got %d", expected, actual)
	}
	for _, sa := range saList1.Items {
		if sa.Namespace != "nsA" {
			t.Fatalf("Expected namespace %q; got %q", "nsA", sa.Namespace)
		}
	}

	saList2, err := tc.CoreV1().ServiceAccounts("nsB").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("ServiceAccounts.List: %s", err)
	}
	if actual, expected := len(saList2.Items), 3; expected != actual {
		t.Fatalf("Expected %d records to match, got %d", expected, actual)
	}
	for _, sa := range saList2.Items {
		if sa.Namespace != "nsB" {
			t.Fatalf("Expected namespace %q; got %q", "nsA", sa.Namespace)
		}
	}

	pod1, err := tc.CoreV1().Pods("nsA").Get(context.TODO(), "pod-1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Pods.Get: %s", err)
	}
	if pod1 == nil {
		t.Fatalf("Expected to find pod nsA/pod-1 but it wasn't found")
	}
	if pod1.Namespace != "nsA" || pod1.Name != "pod-1" {
		t.Fatalf("Expected to find pod nsA/pod-1t, got %s/%s", pod1.Namespace, pod1.Name)
	}

	wrongPod, err := tc.CoreV1().Pods("nsB").Get(context.TODO(), "pod-1", metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Pods.Get: expected nsB/pod-1 not to match, but it matched %s/%s", wrongPod.Namespace, wrongPod.Name)
	}

	allPods, err := tc.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Pods.List: %s", err)
	}
	if actual, expected := len(allPods.Items), 2; expected != actual {
		t.Fatalf("Expected %d pods to match, got %d", expected, actual)
	}

	allSAs, err := tc.CoreV1().ServiceAccounts(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("ServiceAccounts.List: %s", err)
	}
	if actual, expected := len(allSAs.Items), 5; expected != actual {
		t.Fatalf("Expected %d service accounts to match, got %d", expected, actual)
	}
}

func TestFakeClientsetInheritsNamespace(t *testing.T) {
	tc := clientsetfake.NewSimpleClientset(
		testNamespace("nsA"),
		testPod("nsA", "pod-1"),
	)

	_, err := tc.CoreV1().Namespaces().Create(context.TODO(), testNamespace("nsB"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Namespaces.Create: %s", err)
	}

	allNS, err := tc.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Namespaces.List: %s", err)
	}
	if actual, expected := len(allNS.Items), 2; expected != actual {
		t.Fatalf("Expected %d namespaces to match, got %d", expected, actual)
	}

	_, err = tc.CoreV1().Pods("nsB").Create(context.TODO(), testPod("", "pod-1"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Pods.Create nsB/pod-1: %s", err)
	}

	podB1, err := tc.CoreV1().Pods("nsB").Get(context.TODO(), "pod-1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Pods.Get nsB/pod-1: %s", err)
	}
	if podB1 == nil {
		t.Fatalf("Expected to find pod nsB/pod-1 but it wasn't found")
	}
	if podB1.Namespace != "nsB" || podB1.Name != "pod-1" {
		t.Fatalf("Expected to find pod nsB/pod-1t, got %s/%s", podB1.Namespace, podB1.Name)
	}

	_, err = tc.CoreV1().Pods("nsA").Create(context.TODO(), testPod("", "pod-1"), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected Pods.Create to fail with already exists error")
	}

	_, err = tc.CoreV1().Pods("nsA").Update(context.TODO(), testPod("", "pod-1"), metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Pods.Update nsA/pod-1: %s", err)
	}

	_, err = tc.CoreV1().Pods("nsA").Create(context.TODO(), testPod("nsB", "pod-2"), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected Pods.Create to fail with bad request from namespace mismtach")
	}
	if err.Error() != `request namespace does not match object namespace, request: "nsA" object: "nsB"` {
		t.Fatalf("Expected Pods.Create error to provide object and request namespaces, got %q", err)
	}

	_, err = tc.CoreV1().Pods("nsA").Update(context.TODO(), testPod("", "pod-3"), metav1.UpdateOptions{})
	if err == nil {
		t.Fatalf("Expected Pods.Update nsA/pod-3 to fail with not found error")
	}
}

func testSA(ns, name string) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}

func testPod(ns, name string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}

func testNamespace(ns string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns,
		},
	}
}
