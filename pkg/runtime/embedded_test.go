/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/util"
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

func (*ObjectTest) IsAnAPIObject()           {}
func (*ObjectTestExternal) IsAnAPIObject()   {}
func (*EmbeddedTest) IsAnAPIObject()         {}
func (*EmbeddedTestExternal) IsAnAPIObject() {}

func TestDecodeEmptyRawExtensionAsObject(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "v1test"}
	externalGVK := externalGV.WithKind("ObjectTest")

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &ObjectTest{})
	s.AddKnownTypeWithName(externalGVK, &ObjectTestExternal{})

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	obj, _, err := codec.Decode([]byte(`{"kind":"`+externalGVK.Kind+`","apiVersion":"`+externalGV.String()+`","items":[{}]}`), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	test := obj.(*ObjectTest)
	if unk, ok := test.Items[0].(*runtime.Unknown); !ok || unk.Kind != "" || unk.APIVersion != externalGV.String() || string(unk.RawJSON) != "{}" {
		t.Fatalf("unexpected object: %#v", test.Items[0])
	}

	obj, _, err = codec.Decode([]byte(`{"kind":"`+externalGVK.Kind+`","apiVersion":"`+externalGV.String()+`","items":[{"kind":"Other","apiVersion":"v1"}]}`), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	test = obj.(*ObjectTest)
	if unk, ok := test.Items[0].(*runtime.Unknown); !ok || unk.Kind != "Other" || unk.APIVersion != "v1" || string(unk.RawJSON) != `{"kind":"Other","apiVersion":"v1"}` {
		t.Fatalf("unexpected object: %#v", test.Items[0])
	}
}

func TestArrayOfRuntimeObject(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "v1test"}

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &EmbeddedTest{})
	s.AddKnownTypeWithName(externalGV.WithKind("EmbeddedTest"), &EmbeddedTestExternal{})
	s.AddKnownTypes(internalGV, &ObjectTest{})
	s.AddKnownTypeWithName(externalGV.WithKind("ObjectTest"), &ObjectTestExternal{})

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	internal := &ObjectTest{
		Items: []runtime.Object{
			&EmbeddedTest{ID: "foo"},
			&EmbeddedTest{ID: "bar"},
			// TODO: until YAML is removed, this JSON must be in ascending key order to ensure consistent roundtrip serialization
			&runtime.Unknown{RawJSON: []byte(`{"apiVersion":"unknown","foo":"bar","kind":"OtherTest"}`)},
			&ObjectTest{
				Items: []runtime.Object{
					&EmbeddedTest{ID: "baz"},
				},
			},
		},
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
	t.Logf("exact wire is: %s", string(obj.Items[0].RawJSON))

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

	internal.Items[0].(*EmbeddedTest).TypeMeta = runtime.TypeMeta{Kind: "EmbeddedTest", APIVersion: "test.group/v1test"}
	internal.Items[1].(*EmbeddedTest).TypeMeta = runtime.TypeMeta{Kind: "EmbeddedTest", APIVersion: "test.group/v1test"}
	internal.Items[2].(*runtime.Unknown).TypeMeta = runtime.TypeMeta{Kind: "OtherTest", APIVersion: "unknown"}
	internal.Items[3].(*ObjectTest).TypeMeta = runtime.TypeMeta{Kind: "ObjectTest", APIVersion: "test.group/v1test"}
	internal.Items[3].(*ObjectTest).Items[0].(*EmbeddedTest).TypeMeta = runtime.TypeMeta{Kind: "EmbeddedTest", APIVersion: "test.group/v1test"}
	if e, a := internal.Items, list; !reflect.DeepEqual(e, a) {
		t.Errorf("mismatched decoded: %s", util.ObjectGoPrintSideBySide(e, a))
	}
}

func TestEmbeddedObject(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "v1test"}
	embeddedTestExternalGVK := externalGV.WithKind("EmbeddedTest")

	s := runtime.NewScheme()
	s.AddKnownTypes(internalGV, &EmbeddedTest{})
	s.AddKnownTypeWithName(embeddedTestExternalGVK, &EmbeddedTestExternal{})

	outer := &EmbeddedTest{
		ID: "outer",
		Object: &EmbeddedTest{
			ID: "inner",
		},
	}

	codec := serializer.NewCodecFactory(s).LegacyCodec(externalGV)

	wire, err := runtime.Encode(codec, outer)
	if err != nil {
		t.Fatalf("Unexpected encode error '%v'", err)
	}

	t.Logf("Wire format is:\n%v\n", string(wire))

	decoded, err := runtime.Decode(codec, wire)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}

	outer.TypeMeta = runtime.TypeMeta{Kind: "EmbeddedTest", APIVersion: externalGV.String()}
	outer.Object.(*EmbeddedTest).TypeMeta = outer.TypeMeta

	if e, a := outer, decoded; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v but got %#v", e, a)
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
	if !reflect.DeepEqual(externalViaJSON.EmptyObject.RawJSON, []byte("null")) || len(externalViaJSON.Object.RawJSON) == 0 {
		t.Errorf("Expected deserialization of nested objects into bytes, got %#v", externalViaJSON)
	}

	// test JSON decoding, too, since Decode uses yaml unmarshalling.
	// Generic Unmarshalling of JSON cannot load the nested objects because there is
	// no default schema set.  Consumers wishing to get direct JSON decoding must use
	// the external representation
	var decodedViaJSON EmbeddedTest
	err = json.Unmarshal(wire, &decodedViaJSON)
	if err == nil || !strings.Contains(err.Error(), "unmarshal object into Go value of type runtime.Object") {
		t.Fatalf("Unexpected decode error %v", err)
	}
	if a := decodedViaJSON; a.Object != nil || a.EmptyObject != nil {
		t.Errorf("Expected embedded objects to be nil: %#v", a)
	}
}

// TestDeepCopyOfEmbeddedObject checks to make sure that EmbeddedObject's can be passed through DeepCopy with fidelity
func TestDeepCopyOfEmbeddedObject(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "v1test"}
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
