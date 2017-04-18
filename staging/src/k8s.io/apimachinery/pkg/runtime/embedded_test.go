/*
Copyright 2014 The Kubernetes Authors.

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

package runtime_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
)

type EmbeddedTest struct {
	runtime.TypeMeta
	ID          string
	Object      runtime.Object
	EmptyObject runtime.Object
}

type EmbeddedTestExternal struct {
	runtime.TypeMeta `json:",inline"`
	ID               string               `json:"id,omitempty"`
	Object           runtime.RawExtension `json:"object,omitempty"`
	EmptyObject      runtime.RawExtension `json:"emptyObject,omitempty"`
}

type ObjectTest struct {
	runtime.TypeMeta

	ID    string
	Items []runtime.Object
}

type ObjectTestExternal struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`

	ID    string                 `json:"id,omitempty"`
	Items []runtime.RawExtension `json:"items,omitempty"`
}

func (obj *ObjectTest) GetObjectKind() schema.ObjectKind           { return &obj.TypeMeta }
func (obj *ObjectTestExternal) GetObjectKind() schema.ObjectKind   { return &obj.TypeMeta }
func (obj *EmbeddedTest) GetObjectKind() schema.ObjectKind         { return &obj.TypeMeta }
func (obj *EmbeddedTestExternal) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }

func TestDecodeEmptyRawExtensionAsObject(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "v1test"}
	externalGVK := externalGV.WithKind("ObjectTest")

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &ObjectTest{})
	s.AddKnownTypeWithName(externalGVK, &ObjectTestExternal{})

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	obj, gvk, err := codec.Decode([]byte(`{"kind":"`+externalGVK.Kind+`","apiVersion":"`+externalGV.String()+`","items":[{}]}`), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	test := obj.(*ObjectTest)
	if unk, ok := test.Items[0].(*runtime.Unknown); !ok || unk.Kind != "" || unk.APIVersion != "" || string(unk.Raw) != "{}" || unk.ContentType != runtime.ContentTypeJSON {
		t.Fatalf("unexpected object: %#v", test.Items[0])
	}
	if *gvk != externalGVK {
		t.Fatalf("unexpected kind: %#v", gvk)
	}

	obj, gvk, err = codec.Decode([]byte(`{"kind":"`+externalGVK.Kind+`","apiVersion":"`+externalGV.String()+`","items":[{"kind":"Other","apiVersion":"v1"}]}`), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	test = obj.(*ObjectTest)
	if unk, ok := test.Items[0].(*runtime.Unknown); !ok || unk.Kind != "" || unk.APIVersion != "" || string(unk.Raw) != `{"kind":"Other","apiVersion":"v1"}` || unk.ContentType != runtime.ContentTypeJSON {
		t.Fatalf("unexpected object: %#v", test.Items[0])
	}
	if *gvk != externalGVK {
		t.Fatalf("unexpected kind: %#v", gvk)
	}
}

func TestArrayOfRuntimeObject(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "v1test"}

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &EmbeddedTest{})
	s.AddKnownTypeWithName(externalGV.WithKind("EmbeddedTest"), &EmbeddedTestExternal{})
	s.AddKnownTypes(internalGV, &ObjectTest{})
	s.AddKnownTypeWithName(externalGV.WithKind("ObjectTest"), &ObjectTestExternal{})

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	innerItems := []runtime.Object{
		&EmbeddedTest{ID: "baz"},
	}
	items := []runtime.Object{
		&EmbeddedTest{ID: "foo"},
		&EmbeddedTest{ID: "bar"},
		// TODO: until YAML is removed, this JSON must be in ascending key order to ensure consistent roundtrip serialization
		&runtime.Unknown{
			Raw:         []byte(`{"apiVersion":"unknown.group/unknown","foo":"bar","kind":"OtherTest"}`),
			ContentType: runtime.ContentTypeJSON,
		},
		&ObjectTest{
			Items: runtime.NewEncodableList(codec, innerItems),
		},
	}
	internal := &ObjectTest{
		Items: runtime.NewEncodableList(codec, items),
	}
	wire, err := runtime.Encode(codec, internal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("Wire format is:\n%s\n", string(wire))

	obj := &ObjectTestExternal{}
	if err := json.Unmarshal(wire, obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("exact wire is: %s", string(obj.Items[0].Raw))

	items[3] = &ObjectTest{Items: innerItems}
	internal.Items = items

	decoded, err := runtime.Decode(codec, wire)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	list, err := meta.ExtractList(decoded)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if errs := runtime.DecodeList(list, codec); len(errs) > 0 {
		t.Fatalf("unexpected error: %v", errs)
	}

	list2, err := meta.ExtractList(list[3])
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if errs := runtime.DecodeList(list2, codec); len(errs) > 0 {
		t.Fatalf("unexpected error: %v", errs)
	}
	if err := meta.SetList(list[3], list2); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// we want DecodeList to set type meta if possible, even on runtime.Unknown objects
	internal.Items[2].(*runtime.Unknown).TypeMeta = runtime.TypeMeta{Kind: "OtherTest", APIVersion: "unknown.group/unknown"}
	if e, a := internal.Items, list; !reflect.DeepEqual(e, a) {
		t.Errorf("mismatched decoded: %s", diff.ObjectGoPrintSideBySide(e, a))
	}
}

func TestNestedObject(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "v1test"}
	embeddedTestExternalGVK := externalGV.WithKind("EmbeddedTest")

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &EmbeddedTest{})
	s.AddKnownTypeWithName(embeddedTestExternalGVK, &EmbeddedTestExternal{})

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	inner := &EmbeddedTest{
		ID: "inner",
	}
	outer := &EmbeddedTest{
		ID:     "outer",
		Object: runtime.NewEncodable(codec, inner),
	}

	wire, err := runtime.Encode(codec, outer)
	if err != nil {
		t.Fatalf("Unexpected encode error '%v'", err)
	}

	t.Logf("Wire format is:\n%v\n", string(wire))

	decoded, err := runtime.Decode(codec, wire)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}

	// for later tests
	outer.Object = inner

	if e, a := outer, decoded; reflect.DeepEqual(e, a) {
		t.Errorf("Expected unequal %#v %#v", e, a)
	}

	obj, err := runtime.Decode(codec, decoded.(*EmbeddedTest).Object.(*runtime.Unknown).Raw)
	if err != nil {
		t.Fatal(err)
	}
	decoded.(*EmbeddedTest).Object = obj
	if e, a := outer, decoded; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected equal %#v %#v", e, a)
	}

	// test JSON decoding of the external object, which should preserve
	// raw bytes
	var externalViaJSON EmbeddedTestExternal
	err = json.Unmarshal(wire, &externalViaJSON)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}
	if externalViaJSON.Kind == "" || externalViaJSON.APIVersion == "" || externalViaJSON.ID != "outer" {
		t.Errorf("Expected objects to have type info set, got %#v", externalViaJSON)
	}
	if !reflect.DeepEqual(externalViaJSON.EmptyObject.Raw, []byte("null")) || len(externalViaJSON.Object.Raw) == 0 {
		t.Errorf("Expected deserialization of nested objects into bytes, got %#v", externalViaJSON)
	}

	// test JSON decoding, too, since Decode uses yaml unmarshalling.
	// Generic Unmarshalling of JSON cannot load the nested objects because there is
	// no default schema set.  Consumers wishing to get direct JSON decoding must use
	// the external representation
	var decodedViaJSON EmbeddedTest
	err = json.Unmarshal(wire, &decodedViaJSON)
	if err == nil {
		t.Fatal("Expeceted decode error")
	}
	if _, ok := err.(*json.UnmarshalTypeError); !ok {
		t.Fatalf("Unexpected decode error: %v", err)
	}
	if a := decodedViaJSON; a.Object != nil || a.EmptyObject != nil {
		t.Errorf("Expected embedded objects to be nil: %#v", a)
	}
}

// TestDeepCopyOfRuntimeObject checks to make sure that runtime.Objects's can be passed through DeepCopy with fidelity
func TestDeepCopyOfRuntimeObject(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "v1test"}
	embeddedTestExternalGVK := externalGV.WithKind("EmbeddedTest")

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &EmbeddedTest{})
	s.AddKnownTypeWithName(embeddedTestExternalGVK, &EmbeddedTestExternal{})

	original := &EmbeddedTest{
		ID: "outer",
		Object: &EmbeddedTest{
			ID: "inner",
		},
	}

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	originalData, err := runtime.Encode(codec, original)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	t.Logf("originalRole = %v\n", string(originalData))

	copyOfOriginal, err := s.DeepCopy(original)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	copiedData, err := runtime.Encode(codec, copyOfOriginal.(runtime.Object))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	t.Logf("copyOfRole   = %v\n", string(copiedData))

	if !reflect.DeepEqual(original, copyOfOriginal) {
		t.Errorf("expected \n%v\n, got \n%v", string(originalData), string(copiedData))
	}
}
