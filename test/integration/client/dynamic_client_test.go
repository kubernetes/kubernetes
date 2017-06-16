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

package client

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDynamicClient(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	ns := framework.CreateTestingNamespace("dynamic-client", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	gv := &api.Registry.GroupOrDie(v1.GroupName).GroupVersion
	config := &restclient.Config{
		Host:          s.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: gv},
	}

	client := clientset.NewForConfigOrDie(config)
	dynamicClient, err := dynamic.NewClient(config)
	_ = dynamicClient
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}

	// Find the Pod resource
	resources, err := client.Discovery().ServerResourcesForGroupVersion(gv.String())
	if err != nil {
		t.Fatalf("unexpected error listing resources: %v", err)
	}

	var resource metav1.APIResource
	for _, r := range resources.APIResources {
		if r.Kind == "Pod" {
			resource = r
			break
		}
	}

	if len(resource.Name) == 0 {
		t.Fatalf("could not find the pod resource in group/version %q", gv.String())
	}

	// Create a Pod with the normal client
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test",
					Image: "test-image",
				},
			},
		},
	}

	actual, err := client.Core().Pods(ns.Name).Create(pod)
	if err != nil {
		t.Fatalf("unexpected error when creating pod: %v", err)
	}

	// check dynamic list
	obj, err := dynamicClient.Resource(&resource, ns.Name).List(metav1.ListOptions{})
	unstructuredList, ok := obj.(*unstructured.UnstructuredList)
	if !ok {
		t.Fatalf("expected *unstructured.UnstructuredList, got %#v", obj)
	}
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(unstructuredList.Items) != 1 {
		t.Fatalf("expected one pod, got %d", len(unstructuredList.Items))
	}

	got, err := unstructuredToPod(&unstructuredList.Items[0])
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to v1.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// check dynamic get
	unstruct, err := dynamicClient.Resource(&resource, ns.Name).Get(actual.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error when getting pod %q: %v", actual.Name, err)
	}

	got, err = unstructuredToPod(unstruct)
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to v1.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// delete the pod dynamically
	err = dynamicClient.Resource(&resource, ns.Name).Delete(actual.Name, nil)
	if err != nil {
		t.Fatalf("unexpected error when deleting pod: %v", err)
	}

	list, err := client.Core().Pods(ns.Name).List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(list.Items) != 0 {
		t.Fatalf("expected zero pods, got %d", len(list.Items))
	}
}

func unstructuredToPod(obj *unstructured.Unstructured) (*v1.Pod, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	pod := new(v1.Pod)
	err = runtime.DecodeInto(testapi.Default.Codec(), json, pod)
	pod.Kind = ""
	pod.APIVersion = ""
	return pod, err
}
