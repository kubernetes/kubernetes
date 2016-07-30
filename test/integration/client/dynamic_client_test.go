// +build integration,!no-etcd

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	uclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDynamicClient(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("dynamic-client", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	gv := testapi.Default.GroupVersion()
	config := &restclient.Config{
		Host:          s.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: gv},
	}

	client := uclient.NewOrDie(config)
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

	var resource unversioned.APIResource
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
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "test",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "test",
					Image: "test-image",
				},
			},
		},
	}

	actual, err := client.Pods(ns.Name).Create(pod)
	if err != nil {
		t.Fatalf("unexpected error when creating pod: %v", err)
	}

	// check dynamic list
	obj, err := dynamicClient.Resource(&resource, ns.Name).List(&v1.ListOptions{})
	unstructuredList, ok := obj.(*runtime.UnstructuredList)
	if !ok {
		t.Fatalf("expected *runtime.UnstructuredList, got %#v", obj)
	}
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(unstructuredList.Items) != 1 {
		t.Fatalf("expected one pod, got %d", len(unstructuredList.Items))
	}

	got, err := unstructuredToPod(unstructuredList.Items[0])
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to api.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// check dynamic get
	unstruct, err := dynamicClient.Resource(&resource, ns.Name).Get(actual.Name)
	if err != nil {
		t.Fatalf("unexpected error when getting pod %q: %v", actual.Name, err)
	}

	got, err = unstructuredToPod(unstruct)
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to api.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// delete the pod dynamically
	err = dynamicClient.Resource(&resource, ns.Name).Delete(actual.Name, nil)
	if err != nil {
		t.Fatalf("unexpected error when deleting pod: %v", err)
	}

	list, err := client.Pods(ns.Name).List(api.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(list.Items) != 0 {
		t.Fatalf("expected zero pods, got %d", len(list.Items))
	}
}

func unstructuredToPod(obj *runtime.Unstructured) (*api.Pod, error) {
	json, err := runtime.Encode(runtime.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	pod := new(api.Pod)
	err = runtime.DecodeInto(testapi.Default.Codec(), json, pod)
	return pod, err
}
