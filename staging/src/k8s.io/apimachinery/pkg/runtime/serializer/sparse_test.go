/*
Copyright 2018 The Kubernetes Authors.

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

package serializer

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
)

type FakeV1Obj struct {
	metav1.TypeMeta
	metav1.ObjectMeta
}

func (*FakeV1Obj) DeepCopyObject() runtime.Object {
	panic("not supported")
}

type FakeV2DifferentObj struct {
	metav1.TypeMeta
	metav1.ObjectMeta
}

func (*FakeV2DifferentObj) DeepCopyObject() runtime.Object {
	panic("not supported")
}
func TestSparse(t *testing.T) {
	v1 := schema.GroupVersion{Group: "mygroup", Version: "v1"}
	v2 := schema.GroupVersion{Group: "mygroup", Version: "v2"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(v1, &FakeV1Obj{})
	scheme.AddKnownTypes(v2, &FakeV2DifferentObj{})
	codecs := NewCodecFactory(scheme)

	srcObj1 := &FakeV1Obj{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	srcObj2 := &FakeV2DifferentObj{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	encoder := codecs.LegacyCodec(v2, v1)
	decoder := codecs.UniversalDecoder(v2, v1)

	srcObj1Bytes, err := runtime.Encode(encoder, srcObj1)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(srcObj1Bytes))
	srcObj2Bytes, err := runtime.Encode(encoder, srcObj2)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(srcObj2Bytes))

	uncastDstObj1, err := runtime.Decode(decoder, srcObj1Bytes)
	if err != nil {
		t.Fatal(err)
	}
	uncastDstObj2, err := runtime.Decode(decoder, srcObj2Bytes)
	if err != nil {
		t.Fatal(err)
	}

	// clear typemeta
	uncastDstObj1.(*FakeV1Obj).TypeMeta = metav1.TypeMeta{}
	uncastDstObj2.(*FakeV2DifferentObj).TypeMeta = metav1.TypeMeta{}

	if !equality.Semantic.DeepEqual(srcObj1, uncastDstObj1) {
		t.Fatal(diff.ObjectDiff(srcObj1, uncastDstObj1))
	}
	if !equality.Semantic.DeepEqual(srcObj2, uncastDstObj2) {
		t.Fatal(diff.ObjectDiff(srcObj2, uncastDstObj2))
	}
}
