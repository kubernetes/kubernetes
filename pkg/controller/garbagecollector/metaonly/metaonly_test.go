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

	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/runtime/serializer"
)

func getPod() *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
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
		TypeMeta: metav1.TypeMeta{
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

func verfiyMetadata(description string, t *testing.T, in *MetadataOnlyObject) {
	pod := getPod()
	if e, a := pod.ObjectMeta, in.ObjectMeta; !reflect.DeepEqual(e, a) {
		t.Errorf("%s: expected %#v, got %#v", description, e, a)
	}
}

func TestDecodeToMetadataOnlyObject(t *testing.T) {
	data := getPodJson(t)
	cf := serializer.DirectCodecFactory{CodecFactory: NewMetadataCodecFactory()}
	info, ok := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		t.Fatalf("expected to get a JSON serializer")
	}
	codec := cf.DecoderToVersion(info.Serializer, schema.GroupVersion{Group: "SOMEGROUP", Version: "SOMEVERSION"})
	// decode with into
	into := &MetadataOnlyObject{}
	ret, _, err := codec.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	metaOnly, ok := ret.(*MetadataOnlyObject)
	if !ok {
		t.Fatalf("expected ret to be *runtime.MetadataOnlyObject")
	}
	verfiyMetadata("check returned metaonly with into", t, metaOnly)
	verfiyMetadata("check into", t, into)
	// decode without into
	ret, _, err = codec.Decode(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	metaOnly, ok = ret.(*MetadataOnlyObject)
	if !ok {
		t.Fatalf("expected ret to be *runtime.MetadataOnlyObject")
	}
	verfiyMetadata("check returned metaonly without into", t, metaOnly)
}

func verifyListMetadata(t *testing.T, metaOnlyList *MetadataOnlyObjectList) {
	items, err := meta.ExtractList(metaOnlyList)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		metaOnly, ok := item.(*MetadataOnlyObject)
		if !ok {
			t.Fatalf("expected item to be *MetadataOnlyObject")
		}
		verfiyMetadata("check list", t, metaOnly)
	}
}

func TestDecodeToMetadataOnlyObjectList(t *testing.T) {
	data := getPodListJson(t)
	cf := serializer.DirectCodecFactory{CodecFactory: NewMetadataCodecFactory()}
	info, ok := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		t.Fatalf("expected to get a JSON serializer")
	}
	codec := cf.DecoderToVersion(info.Serializer, schema.GroupVersion{Group: "SOMEGROUP", Version: "SOMEVERSION"})
	// decode with into
	into := &MetadataOnlyObjectList{}
	ret, _, err := codec.Decode(data, nil, into)
	if err != nil {
		t.Fatal(err)
	}
	metaOnlyList, ok := ret.(*MetadataOnlyObjectList)
	if !ok {
		t.Fatalf("expected ret to be *runtime.UnstructuredList")
	}
	verifyListMetadata(t, metaOnlyList)
	verifyListMetadata(t, into)
	// decode without into
	ret, _, err = codec.Decode(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	metaOnlyList, ok = ret.(*MetadataOnlyObjectList)
	if !ok {
		t.Fatalf("expected ret to be *runtime.UnstructuredList")
	}
	verifyListMetadata(t, metaOnlyList)
}
