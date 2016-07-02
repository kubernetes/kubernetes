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

package garbagecollector

import (
	"encoding/json"
	"testing"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

func getPod() *v1.Pod {
	return &v1.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: "pod",
			OwnerReferences: []v1.OwnerReference{
				{UID: "1234"},
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
	}
}

func getPodJson(t *testing.T) []byte {
	data, err := json.Marshal(getPod())
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func getPodListJson(t *testing.T) []byte {
	data, err := json.Marshal(&v1.PodList{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "PodList",
			APIVersion: "v1",
		},
		Items: []v1.Pod{
			*getPod(),
			*getPod(),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func verifyMetadataKeys(t *testing.T, in map[string]interface{}) {
	if _, ok := in["apiVersion"]; !ok {
		t.Errorf("mising apiVersion")
	}
	if _, ok := in["kind"]; !ok {
		t.Errorf("mising kind")
	}
	if _, ok := in["metadata"]; !ok {
		t.Errorf("mising metadata")
	}
	if len(in) != 3 {
		t.Errorf("got unexpected keys: %#v", in)
	}
}

func TestDecodeToUnstructured(t *testing.T) {
	data := getPodJson(t)
	codec := NewCompressingCodec()
	into := &runtime.Unstructured{}
	ret, _, err := codec.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	unstructured, ok := ret.(*runtime.Unstructured)
	if !ok {
		t.Fatalf("expect ret to be *runtime.Unstructured")
	}
	verifyMetadataKeys(t, unstructured.Object)
	verifyMetadataKeys(t, into.Object)
}

func verifyUnstructureListMetadataKeys(t *testing.T, unstructuredList *runtime.UnstructuredList) {
	// this is how reflector handles the unstructuredList
	items, err := meta.ExtractList(unstructuredList)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		unstructured, ok := item.(*runtime.Unstructured)
		if !ok {
			t.Fatalf("expect item to be *runtime.Unstructured")
		}
		verifyMetadataKeys(t, unstructured.Object)
	}
}

func TestDecodeToUnstructuredList(t *testing.T) {
	data := getPodListJson(t)
	into := &runtime.UnstructuredList{}
	codec := NewCompressingCodec()
	ret, _, err := codec.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	unstructuredList, ok := ret.(*runtime.UnstructuredList)
	if !ok {
		t.Fatalf("expect ret to be *runtime.UnstructuredList")
	}
	verifyUnstructureListMetadataKeys(t, unstructuredList)
	verifyUnstructureListMetadataKeys(t, into)
}
