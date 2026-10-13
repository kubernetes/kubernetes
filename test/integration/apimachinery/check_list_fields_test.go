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

package apimachinery

import (
	"context"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCheckFieldItemsInEmptyList(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--runtime-config=api/all=true",
	}, framework.SharedEtcd())

	defer server.TearDownFn()

	clientSet, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	_, lists, err := clientSet.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatal(err)
	}

	for _, resources := range lists {
		for _, resource := range resources.APIResources {
			gv, err := schema.ParseGroupVersion(resources.GroupVersion)
			if err != nil {
				t.Fatal(err)
			}
			gvr := schema.GroupVersionResource{
				Group:    gv.Group,
				Version:  gv.Version,
				Resource: resource.Name,
			}

			if !sets.NewString(resource.Verbs...).Has("list") {
				t.Logf("skip gvr: %s", gvr)
				continue
			}

			var list *unstructured.UnstructuredList
			if resource.Namespaced {
				list, err = dynamicClient.Resource(gvr).Namespace("non-exist").List(context.Background(),
					metav1.ListOptions{LabelSelector: "non-exist=non-exist"})
			} else {
				list, err = dynamicClient.Resource(gvr).List(context.Background(),
					metav1.ListOptions{LabelSelector: "non-exist=non-exist"})
			}

			if err != nil {
				t.Fatal(err)
			}

			if list == nil {
				t.Fatalf("gvr: %s, list is nil", gvr)
			}

			// Field `items` in List object should be a zero-length array when no obj in etcd.
			if list.Items == nil {
				t.Fatalf("gvr: %s, fields items of list is nil", gvr)
			}

			if len(list.Items) > 0 {
				t.Fatalf("gvr: %s, fields items should be a zero-length array", gvr)
			}
		}
	}
}
