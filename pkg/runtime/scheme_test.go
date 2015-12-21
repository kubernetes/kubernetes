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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/util"
)

type TypeMeta struct {
	Kind       string `json:"kind,omitempty"`
	APIVersion string `json:"apiVersion,omitempty"`
}

// SetGroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) GroupVersionKind() *unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

type InternalSimple struct {
	TypeMeta   `json:",inline"`
	TestString string `json:"testString"`
}

type ExternalSimple struct {
	TypeMeta   `json:",inline"`
	TestString string `json:"testString"`
}

func (obj *InternalSimple) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *InternalSimple) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	obj.TypeMeta.APIVersion, obj.TypeMeta.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalSimple) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *ExternalSimple) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	obj.TypeMeta.APIVersion, obj.TypeMeta.Kind = gvk.ToAPIVersionAndKind()
}

func TestScheme(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})

	// test that scheme is an ObjectTyper
	var _ runtime.ObjectTyper = scheme

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := scheme.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			if e, a := internalGV.String(), scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := externalGV.String(), scope.Meta().DestVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *ExternalSimple, out *InternalSimple, scope conversion.Scope) error {
			if e, a := externalGV.String(), scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := internalGV.String(), scope.Meta().DestVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			scope.Convert(&in.TestString, &out.TestString, 0)
			externalToInternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	codecs := serializer.NewCodecFactory(scheme)
	codec := codecs.LegacyCodec(externalGV)
	jsonserializer, _ := codecs.SerializerForFileExtension("json")

	simple := &InternalSimple{
		TestString: "foo",
	}

	// Test Encode, Decode, DecodeInto, and DecodeToVersion
	obj := runtime.Object(simple)
	data, err := runtime.Encode(codec, obj)
	obj2, err2 := runtime.Decode(codec, data)
	obj3 := &InternalSimple{}
	err3 := runtime.DecodeInto(codec, data, obj3)
	obj4, err4 := runtime.Decode(jsonserializer, data)
	if err != nil || err2 != nil || err3 != nil || err4 != nil {
		t.Fatalf("Failure: '%v' '%v' '%v' '%v'", err, err2, err3, err4)
	}
	if _, ok := obj2.(*InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if e, a := simple, obj2; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}
	simple.TypeMeta = TypeMeta{Kind: "Simple", APIVersion: externalGV.String()}
	if e, a := simple, obj3; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}
	if _, ok := obj4.(*ExternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}

	// Test Convert
	external := &ExternalSimple{}
	err = scheme.Convert(simple, external)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, external.TestString; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Encode and Convert should each have caused an increment.
	if e, a := 2, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	// DecodeInto and Decode should each have caused an increment because of a conversion
	if e, a := 2, externalToInternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestBadJSONRejection(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	jsonserializer, _ := codecs.SerializerForFileExtension("json")

	badJSONMissingKind := []byte(`{ }`)
	if _, err := runtime.Decode(jsonserializer, badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := runtime.Decode(jsonserializer, badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}

type ExtensionA struct {
	TypeMeta   `json:",inline"`
	TestString string `json:"testString"`
}

type ExtensionB struct {
	TypeMeta   `json:",inline"`
	TestString string `json:"testString"`
}

type ExternalExtensionType struct {
	TypeMeta  `json:",inline"`
	Extension runtime.RawExtension `json:"extension"`
}

type InternalExtensionType struct {
	TypeMeta  `json:",inline"`
	Extension runtime.Object `json:"extension"`
}

type ExternalOptionalExtensionType struct {
	TypeMeta  `json:",inline"`
	Extension runtime.RawExtension `json:"extension,omitempty"`
}

type InternalOptionalExtensionType struct {
	TypeMeta  `json:",inline"`
	Extension runtime.Object `json:"extension,omitempty"`
}

func (obj *ExtensionA) GetObjectKind() unversioned.ObjectKind                    { return &obj.TypeMeta }
func (obj *ExtensionB) GetObjectKind() unversioned.ObjectKind                    { return &obj.TypeMeta }
func (obj *ExternalExtensionType) GetObjectKind() unversioned.ObjectKind         { return &obj.TypeMeta }
func (obj *InternalExtensionType) GetObjectKind() unversioned.ObjectKind         { return &obj.TypeMeta }
func (obj *ExternalOptionalExtensionType) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *InternalOptionalExtensionType) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }

func TestExternalToInternalMapping(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("OptionalExtensionType"), &InternalOptionalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("OptionalExtensionType"), &ExternalOptionalExtensionType{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	table := []struct {
		obj     runtime.Object
		encoded string
	}{
		{
			&InternalOptionalExtensionType{Extension: nil},
			`{"kind":"OptionalExtensionType","apiVersion":"` + externalGV.String() + `"}`,
		},
	}

	for i, item := range table {
		gotDecoded, err := runtime.Decode(codec, []byte(item.encoded))
		if err != nil {
			t.Errorf("unexpected error '%v' (%v)", err, item.encoded)
		} else if e, a := item.obj, gotDecoded; !reflect.DeepEqual(e, a) {
			t.Errorf("%d: unexpected objects:\n%s", i, util.ObjectGoPrintSideBySide(e, a))
		}
	}
}

func TestExtensionMapping(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("ExtensionType"), &InternalExtensionType{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("OptionalExtensionType"), &InternalOptionalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("ExtensionType"), &ExternalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("OptionalExtensionType"), &ExternalOptionalExtensionType{})

	// register external first when the object is the same in both schemes, so ObjectVersionAndKind reports the
	// external version.
	scheme.AddKnownTypeWithName(externalGV.WithKind("A"), &ExtensionA{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("B"), &ExtensionB{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("A"), &ExtensionA{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("B"), &ExtensionB{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	table := []struct {
		obj      runtime.Object
		expected runtime.Object
		encoded  string
	}{
		{
			&InternalExtensionType{
				Extension: runtime.NewEncodable(codec, &ExtensionA{TestString: "foo"}),
			},
			&InternalExtensionType{
				Extension: &runtime.Unknown{
					RawJSON: []byte(`{"kind":"A","apiVersion":"test.group/testExternal","testString":"foo"}`),
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"kind":"ExtensionType","apiVersion":"` + externalGV.String() + `","extension":{"kind":"A","apiVersion":"test.group/testExternal","testString":"foo"}}
`,
		}, {
			&InternalExtensionType{Extension: runtime.NewEncodable(codec, &ExtensionB{TestString: "bar"})},
			&InternalExtensionType{
				Extension: &runtime.Unknown{
					RawJSON: []byte(`{"kind":"B","apiVersion":"test.group/testExternal","testString":"bar"}`),
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"kind":"ExtensionType","apiVersion":"` + externalGV.String() + `","extension":{"kind":"B","apiVersion":"test.group/testExternal","testString":"bar"}}
`,
		}, {
			&InternalExtensionType{Extension: nil},
			&InternalExtensionType{
				Extension: nil,
			},
			`{"kind":"ExtensionType","apiVersion":"` + externalGV.String() + `","extension":null}
`,
		},
	}

	for i, item := range table {
		gotEncoded, err := runtime.Encode(codec, item.obj)
		if err != nil {
			t.Errorf("unexpected error '%v' (%#v)", err, item.obj)
		} else if e, a := item.encoded, string(gotEncoded); e != a {
			t.Errorf("expected\n%#v\ngot\n%#v\n", e, a)
		}

		gotDecoded, err := runtime.Decode(codec, []byte(item.encoded))
		if err != nil {
			t.Errorf("unexpected error '%v' (%v)", err, item.encoded)
		} else if e, a := item.expected, gotDecoded; !reflect.DeepEqual(e, a) {
			t.Errorf("%d: unexpected objects:\n%s", i, util.ObjectGoPrintSideBySide(e, a))
		}
	}
}

func TestEncode(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	test := &InternalSimple{
		TestString: "I'm the same",
	}
	obj := runtime.Object(test)
	data, err := runtime.Encode(codec, obj)
	obj2, gvk, err2 := codec.Decode(data, nil, nil)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, test) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", test, obj2)
	}
	if !reflect.DeepEqual(gvk, &unversioned.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "Simple"}) {
		t.Errorf("unexpected gvk returned by decode: %#v", gvk)
	}
}
