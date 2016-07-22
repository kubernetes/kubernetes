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

package metaonly

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
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

func verfiyMetadata(t *testing.T, in *MetaOnly) {
	pod := getPod()
	if e, a := pod.ObjectMeta, in.ObjectMeta; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %#v, got %#v", e, a)
	}
	if e, a := (unversioned.TypeMeta{APIVersion: "v1", Kind: "Pod"}), in.TypeMeta; e != a {
		t.Errorf("expect %#v, got %#v", e, a)
	}
}

func TestDecodeToMetaOnly(t *testing.T) {
	data := getPodJson(t)
	into := &MetaOnly{}
	ret, _, err := MetaOnlyJSONScheme.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	metaOnly, ok := ret.(*MetaOnly)
	if !ok {
		t.Fatalf("expect ret to be *runtime.MetaOnly")
	}
	verfiyMetadata(t, metaOnly)
	verfiyMetadata(t, into)
}

func verifyListMetadata(t *testing.T, metaOnlyList *MetaOnlyList) {
	items, err := meta.ExtractList(metaOnlyList)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		metaOnly, ok := item.(*MetaOnly)
		if !ok {
			t.Fatalf("expect item to be *MetaOnly")
		}
		verfiyMetadata(t, metaOnly)
	}
}

func TestDecodeToMetaOnlyList(t *testing.T) {
	data := getPodListJson(t)
	into := &MetaOnlyList{}
	ret, _, err := MetaOnlyJSONScheme.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	metaOnlyList, ok := ret.(*MetaOnlyList)
	if !ok {
		t.Fatalf("expect ret to be *runtime.UnstructuredList")
	}
	verifyListMetadata(t, metaOnlyList)
	verifyListMetadata(t, into)
}
