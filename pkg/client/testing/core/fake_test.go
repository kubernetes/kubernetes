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

package core_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	clientsetfake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
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

	saList1, err := tc.Core().ServiceAccounts("nsA").List(api.ListOptions{})
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

	saList2, err := tc.Core().ServiceAccounts("nsB").List(api.ListOptions{})
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

	pod1, err := tc.Core().Pods("nsA").Get("pod-1")
	if err != nil {
		t.Fatalf("Pods.Get: %s", err)
	}
	if pod1 == nil {
		t.Fatalf("Expected to find pod nsA/pod-1 but it wasn't found")
	}
	if pod1.Namespace != "nsA" || pod1.Name != "pod-1" {
		t.Fatalf("Expected to find pod nsA/pod-1t, got %s/%s", pod1.Namespace, pod1.Name)
	}

	wrongPod, err := tc.Core().Pods("nsB").Get("pod-1")
	if err == nil {
		t.Fatalf("Pods.Get: expected nsB/pod-1 not to match, but it matched %s/%s", wrongPod.Namespace, wrongPod.Name)
	}

	allPods, err := tc.Core().Pods(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		t.Fatalf("Pods.List: %s", err)
	}
	if actual, expected := len(allPods.Items), 2; expected != actual {
		t.Fatalf("Expected %d pods to match, got %d", expected, actual)
	}

	allSAs, err := tc.Core().ServiceAccounts(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		t.Fatalf("ServiceAccounts.List: %s", err)
	}
	if actual, expected := len(allSAs.Items), 5; expected != actual {
		t.Fatalf("Expected %d service accounts to match, got %d", expected, actual)
	}
}

func testSA(ns, name string) *api.ServiceAccount {
	return &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}

func testPod(ns, name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}
