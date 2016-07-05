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
	"reflect"
	"testing"

	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/util/diff"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

type InternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

func (obj *InternalSimple) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *ExternalSimple) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }

func TestScheme(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})

	// If set, would clear TypeMeta during conversion.
	//scheme.AddIgnoredConversionType(&TypeMeta{}, &TypeMeta{})

	// test that scheme is an ObjectTyper
	var _ runtime.ObjectTyper = scheme

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := scheme.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			scope.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *ExternalSimple, out *InternalSimple, scope conversion.Scope) error {
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
	if err != nil {
		t.Fatal(err)
	}

	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := obj2.(*InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if e, a := simple, obj2; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}

	obj3 := &InternalSimple{}
	if err := runtime.DecodeInto(codec, data, obj3); err != nil {
		t.Fatal(err)
	}
	// clearing TypeMeta is a function of the scheme, which we do not test here (ConvertToVersion
	// does not automatically clear TypeMeta anymore).
	simple.TypeMeta = runtime.TypeMeta{Kind: "Simple", APIVersion: externalGV.String()}
	if e, a := simple, obj3; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}

	obj4, err := runtime.Decode(jsonserializer, data)
	if err != nil {
		t.Fatal(err)
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
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExtensionB struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension"`
}

type InternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension"`
}

type ExternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension,omitempty"`
}

type InternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension,omitempty"`
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
			t.Errorf("%d: unexpected objects:\n%s", i, diff.ObjectGoPrintSideBySide(e, a))
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
					Raw:         []byte(`{"apiVersion":"test.group/testExternal","kind":"A","testString":"foo"}`),
					ContentType: runtime.ContentTypeJSON,
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"apiVersion":"` + externalGV.String() + `","kind":"ExtensionType","extension":{"apiVersion":"test.group/testExternal","kind":"A","testString":"foo"}}
`,
		}, {
			&InternalExtensionType{Extension: runtime.NewEncodable(codec, &ExtensionB{TestString: "bar"})},
			&InternalExtensionType{
				Extension: &runtime.Unknown{
					Raw:         []byte(`{"apiVersion":"test.group/testExternal","kind":"B","testString":"bar"}`),
					ContentType: runtime.ContentTypeJSON,
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"apiVersion":"` + externalGV.String() + `","kind":"ExtensionType","extension":{"apiVersion":"test.group/testExternal","kind":"B","testString":"bar"}}
`,
		}, {
			&InternalExtensionType{Extension: nil},
			&InternalExtensionType{
				Extension: nil,
			},
			`{"apiVersion":"` + externalGV.String() + `","kind":"ExtensionType","extension":null}
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
			t.Errorf("%d: unexpected objects:\n%s", i, diff.ObjectGoPrintSideBySide(e, a))
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

func TestUnversionedTypes(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "testExternal"}
	otherGV := unversioned.GroupVersion{Group: "group", Version: "other"}

	scheme := runtime.NewScheme()
	scheme.AddUnversionedTypes(externalGV, &InternalSimple{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})
	scheme.AddKnownTypeWithName(otherGV.WithKind("Simple"), &ExternalSimple{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	if unv, ok := scheme.IsUnversioned(&InternalSimple{}); !unv || !ok {
		t.Fatalf("type not unversioned and in scheme: %t %t", unv, ok)
	}

	kinds, _, err := scheme.ObjectKinds(&InternalSimple{})
	if err != nil {
		t.Fatal(err)
	}
	kind := kinds[0]
	if kind != externalGV.WithKind("InternalSimple") {
		t.Fatalf("unexpected: %#v", kind)
	}

	test := &InternalSimple{
		TestString: "I'm the same",
	}
	obj := runtime.Object(test)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Fatal(err)
	}
	obj2, gvk, err := codec.Decode(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := obj2.(*InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, test) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", test, obj2)
	}
	// object is serialized as an unversioned object (in the group and version it was defined in)
	if !reflect.DeepEqual(gvk, &unversioned.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "InternalSimple"}) {
		t.Errorf("unexpected gvk returned by decode: %#v", gvk)
	}

	// when serialized to a different group, the object is kept in its preferred name
	codec = serializer.NewCodecFactory(scheme).LegacyCodec(otherGV)
	data, err = runtime.Encode(codec, obj)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"apiVersion":"test.group/testExternal","kind":"InternalSimple","testString":"I'm the same"}`+"\n" {
		t.Errorf("unexpected data: %s", data)
	}
}

// Test a weird version/kind embedding format.
type MyWeirdCustomEmbeddedVersionKindField struct {
	ID         string `json:"ID,omitempty"`
	APIVersion string `json:"myVersionKey,omitempty"`
	ObjectKind string `json:"myKindKey,omitempty"`
	Z          string `json:"Z,omitempty"`
	Y          uint64 `json:"Y,omitempty"`
}

type TestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string               `json:"A,omitempty"`
	B                                     int                  `json:"B,omitempty"`
	C                                     int8                 `json:"C,omitempty"`
	D                                     int16                `json:"D,omitempty"`
	E                                     int32                `json:"E,omitempty"`
	F                                     int64                `json:"F,omitempty"`
	G                                     uint                 `json:"G,omitempty"`
	H                                     uint8                `json:"H,omitempty"`
	I                                     uint16               `json:"I,omitempty"`
	J                                     uint32               `json:"J,omitempty"`
	K                                     uint64               `json:"K,omitempty"`
	L                                     bool                 `json:"L,omitempty"`
	M                                     map[string]int       `json:"M,omitempty"`
	N                                     map[string]TestType2 `json:"N,omitempty"`
	O                                     *TestType2           `json:"O,omitempty"`
	P                                     []TestType2          `json:"Q,omitempty"`
}

type TestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

type ExternalTestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}
type ExternalTestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string                       `json:"A,omitempty"`
	B                                     int                          `json:"B,omitempty"`
	C                                     int8                         `json:"C,omitempty"`
	D                                     int16                        `json:"D,omitempty"`
	E                                     int32                        `json:"E,omitempty"`
	F                                     int64                        `json:"F,omitempty"`
	G                                     uint                         `json:"G,omitempty"`
	H                                     uint8                        `json:"H,omitempty"`
	I                                     uint16                       `json:"I,omitempty"`
	J                                     uint32                       `json:"J,omitempty"`
	K                                     uint64                       `json:"K,omitempty"`
	L                                     bool                         `json:"L,omitempty"`
	M                                     map[string]int               `json:"M,omitempty"`
	N                                     map[string]ExternalTestType2 `json:"N,omitempty"`
	O                                     *ExternalTestType2           `json:"O,omitempty"`
	P                                     []ExternalTestType2          `json:"Q,omitempty"`
}

type ExternalInternalSame struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     TestType2 `json:"A,omitempty"`
}

func (obj *MyWeirdCustomEmbeddedVersionKindField) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *MyWeirdCustomEmbeddedVersionKindField) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.ObjectKind = gvk.ToAPIVersionAndKind()
}
func (obj *MyWeirdCustomEmbeddedVersionKindField) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.ObjectKind)
}

func (obj *ExternalInternalSame) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType1) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *ExternalTestType1) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType2) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }
func (obj *ExternalTestType2) GetObjectKind() unversioned.ObjectKind {
	return unversioned.EmptyObjectKind
}

// TestObjectFuzzer can randomly populate all the above objects.
var TestObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 100).Funcs(
	func(j *MyWeirdCustomEmbeddedVersionKindField, c fuzz.Continue) {
		// We have to customize the randomization of MyWeirdCustomEmbeddedVersionKindFields because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.ObjectKind = ""
		j.ID = c.RandString()
	},
)

// Returns a new Scheme set up with the test objects.
func GetTestScheme() *runtime.Scheme {
	internalGV := unversioned.GroupVersion{Version: "__internal"}
	externalGV := unversioned.GroupVersion{Version: "v1"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &TestType1{}, &TestType2{}, &ExternalInternalSame{})
	s.AddKnownTypes(externalGV, &ExternalInternalSame{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType2"), &ExternalTestType2{})
	s.AddKnownTypeWithName(internalGV.WithKind("TestType3"), &TestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType3"), &ExternalTestType1{})
	return s
}

func TestKnownTypes(t *testing.T) {
	s := GetTestScheme()
	if len(s.KnownTypes(unversioned.GroupVersion{Group: "group", Version: "v2"})) != 0 {
		t.Errorf("should have no known types for v2")
	}

	types := s.KnownTypes(unversioned.GroupVersion{Version: "v1"})
	for _, s := range []string{"TestType1", "TestType2", "TestType3", "ExternalInternalSame"} {
		if _, ok := types[s]; !ok {
			t.Errorf("missing type %q", s)
		}
	}
}

func TestConvertToVersion(t *testing.T) {
	s := GetTestScheme()
	tt := &TestType1{A: "I'm not a pointer object"}
	other, err := s.ConvertToVersion(tt, unversioned.GroupVersion{Version: "v1"})
	if err != nil {
		t.Fatalf("Failure: %v", err)
	}
	converted, ok := other.(*ExternalTestType1)
	if !ok {
		t.Fatalf("Got wrong type")
	}
	if tt.A != converted.A {
		t.Fatalf("Failed to convert object correctly: %#v", converted)
	}
}

func TestMetaValues(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: "__internal"}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "externalVersion"}

	s := runtime.NewScheme()
	s.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	s.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			t.Logf("internal -> external")
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *ExternalSimple, out *InternalSimple, scope conversion.Scope) error {
			t.Logf("external -> internal")
			scope.Convert(&in.TestString, &out.TestString, 0)
			externalToInternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	simple := &InternalSimple{
		TestString: "foo",
	}

	s.Log(t)

	out, err := s.ConvertToVersion(simple, externalGV)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	internal, err := s.ConvertToVersion(out, internalGV)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e, a := simple, internal; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}

	if e, a := 1, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := 1, externalToInternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestMetaValuesUnregisteredConvert(t *testing.T) {
	type InternalSimple struct {
		Version    string `json:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty"`
		TestString string `json:"testString"`
	}
	type ExternalSimple struct {
		Version    string `json:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty"`
		TestString string `json:"testString"`
	}
	s := runtime.NewScheme()
	// We deliberately don't register the types.

	internalToExternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	simple := &InternalSimple{TestString: "foo"}
	external := &ExternalSimple{}
	err = s.Convert(simple, external)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, external.TestString; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Verify that our conversion handler got called.
	if e, a := 1, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}
