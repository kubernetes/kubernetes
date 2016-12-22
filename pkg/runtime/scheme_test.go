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
	"strings"
	"testing"

	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"

	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
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

func (obj *InternalSimple) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
func (obj *ExternalSimple) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }

func TestScheme(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

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
	info, _ := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	jsonserializer := info.Serializer

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
	err = scheme.Convert(simple, external, nil)
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
	info, _ := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	jsonserializer := info.Serializer

	badJSONMissingKind := []byte(`{ }`)
	if _, err := runtime.Decode(jsonserializer, badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := runtime.Decode(jsonserializer, badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Node{}); err2 == nil {
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

func (obj *ExtensionA) GetObjectKind() schema.ObjectKind                    { return &obj.TypeMeta }
func (obj *ExtensionB) GetObjectKind() schema.ObjectKind                    { return &obj.TypeMeta }
func (obj *ExternalExtensionType) GetObjectKind() schema.ObjectKind         { return &obj.TypeMeta }
func (obj *InternalExtensionType) GetObjectKind() schema.ObjectKind         { return &obj.TypeMeta }
func (obj *ExternalOptionalExtensionType) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
func (obj *InternalOptionalExtensionType) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }

func TestExternalToInternalMapping(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

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
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

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
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

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
	if !reflect.DeepEqual(gvk, &schema.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "Simple"}) {
		t.Errorf("unexpected gvk returned by decode: %#v", gvk)
	}
}

func TestUnversionedTypes(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}
	otherGV := schema.GroupVersion{Group: "group", Version: "other"}

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
	if !reflect.DeepEqual(gvk, &schema.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "InternalSimple"}) {
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

type UnversionedType struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string `json:"A,omitempty"`
}

type UnknownType struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string `json:"A,omitempty"`
}

func (obj *MyWeirdCustomEmbeddedVersionKindField) GetObjectKind() schema.ObjectKind { return obj }
func (obj *MyWeirdCustomEmbeddedVersionKindField) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.ObjectKind = gvk.ToAPIVersionAndKind()
}
func (obj *MyWeirdCustomEmbeddedVersionKindField) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.ObjectKind)
}

func (obj *ExternalInternalSame) GetObjectKind() schema.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType1) GetObjectKind() schema.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *ExternalTestType1) GetObjectKind() schema.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType2) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *ExternalTestType2) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
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
	internalGV := schema.GroupVersion{Version: "__internal"}
	externalGV := schema.GroupVersion{Version: "v1"}
	alternateExternalGV := schema.GroupVersion{Group: "custom", Version: "v1"}
	differentExternalGV := schema.GroupVersion{Group: "other", Version: "v2"}

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
	s.AddKnownTypeWithName(externalGV.WithKind("TestType4"), &ExternalTestType1{})
	s.AddKnownTypeWithName(alternateExternalGV.WithKind("TestType3"), &ExternalTestType1{})
	s.AddKnownTypeWithName(alternateExternalGV.WithKind("TestType5"), &ExternalTestType1{})
	s.AddKnownTypeWithName(differentExternalGV.WithKind("TestType1"), &ExternalTestType1{})
	s.AddUnversionedTypes(externalGV, &UnversionedType{})
	return s
}

func TestKnownTypes(t *testing.T) {
	s := GetTestScheme()
	if len(s.KnownTypes(schema.GroupVersion{Group: "group", Version: "v2"})) != 0 {
		t.Errorf("should have no known types for v2")
	}

	types := s.KnownTypes(schema.GroupVersion{Version: "v1"})
	for _, s := range []string{"TestType1", "TestType2", "TestType3", "ExternalInternalSame"} {
		if _, ok := types[s]; !ok {
			t.Errorf("missing type %q", s)
		}
	}
}

func TestConvertToVersionBasic(t *testing.T) {
	s := GetTestScheme()
	tt := &TestType1{A: "I'm not a pointer object"}
	other, err := s.ConvertToVersion(tt, schema.GroupVersion{Version: "v1"})
	if err != nil {
		t.Fatalf("Failure: %v", err)
	}
	converted, ok := other.(*ExternalTestType1)
	if !ok {
		t.Fatalf("Got wrong type: %T", other)
	}
	if tt.A != converted.A {
		t.Fatalf("Failed to convert object correctly: %#v", converted)
	}
}

type testGroupVersioner struct {
	target schema.GroupVersionKind
	ok     bool
}

func (m testGroupVersioner) KindForGroupVersionKinds(kinds []schema.GroupVersionKind) (schema.GroupVersionKind, bool) {
	return m.target, m.ok
}

func TestConvertToVersion(t *testing.T) {
	testCases := []struct {
		scheme *runtime.Scheme
		in     runtime.Object
		gv     runtime.GroupVersioner
		same   bool
		out    runtime.Object
		errFn  func(error) bool
	}{
		// errors if the type is not registered in the scheme
		{
			scheme: GetTestScheme(),
			in:     &UnknownType{},
			errFn:  func(err error) bool { return err != nil && runtime.IsNotRegisteredError(err) },
		},
		// errors if the group versioner returns no target
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{},
			errFn: func(err error) bool {
				return err != nil && strings.Contains(err.Error(), "is not suitable for converting")
			},
		},
		// converts to internal
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     schema.GroupVersion{Version: "__internal"},
			out:    &TestType1{A: "test"},
		},
		// prefers the best match
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "__internal"}, {Version: "v1"}},
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// unversioned type returned as-is
		{
			scheme: GetTestScheme(),
			in:     &UnversionedType{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}},
			same:   true,
			out: &UnversionedType{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "UnversionedType"},
				A: "test",
			},
		},
		// unversioned type returned when not included in the target types
		{
			scheme: GetTestScheme(),
			in:     &UnversionedType{A: "test"},
			gv:     schema.GroupVersions{{Group: "other", Version: "v2"}},
			same:   true,
			out: &UnversionedType{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "UnversionedType"},
				A: "test",
			},
		},
		// detected as already being in the target version
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// detected as already being in the first target version
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}, {Version: "__internal"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// detected as already being in the first target version
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}, {Version: "__internal"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (1/3): different kind
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType3", Version: "v1"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (2/3): different gv
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType3", Group: "custom", Version: "v1"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (3/3): different gvk
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Group: "custom", Version: "v1", Kind: "TestType5"}},
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType5"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, copies because version differs
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "other", Version: "v2"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}, schema.GroupKind{Kind: "TestType1"}),
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "other/v2", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, copies because version differs
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "other", Version: "v2"}, schema.GroupKind{Kind: "TestType1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}),
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "other/v2", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner is unable to find a match when kind AND group don't match (there is no TestType1 kind in group "other", and no kind "TestType5" in the default group)
		{
			scheme: GetTestScheme(),
			in:     &TestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "custom", Version: "v1"}, schema.GroupKind{Group: "other"}, schema.GroupKind{Kind: "TestType5"}),
			errFn: func(err error) bool {
				return err != nil && strings.Contains(err.Error(), "is not suitable for converting")
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, performs no copy
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "", Version: "v1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}, schema.GroupKind{Kind: "TestType1"}),
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, performs no copy
		{
			scheme: GetTestScheme(),
			in:     &ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "", Version: "v1"}, schema.GroupKind{Kind: "TestType1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}),
			same:   true,
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// group versioner can choose a particular target kind for a given input when kind is the same across group versions
		{
			scheme: GetTestScheme(),
			in:     &TestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Version: "v1", Kind: "TestType3"}},
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// group versioner can choose a different kind
		{
			scheme: GetTestScheme(),
			in:     &TestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType5", Group: "custom", Version: "v1"}},
			out: &ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType5"},
				A: "test",
			},
		},
	}
	for i, test := range testCases {
		original, _ := test.scheme.DeepCopy(test.in)
		out, err := test.scheme.ConvertToVersion(test.in, test.gv)
		switch {
		case test.errFn != nil:
			if !test.errFn(err) {
				t.Errorf("%d: unexpected error: %v", i, err)
			}
			continue
		case err != nil:
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if out == test.in {
			t.Errorf("%d: ConvertToVersion should always copy out: %#v", i, out)
			continue
		}

		if test.same {
			if !reflect.DeepEqual(original, test.in) {
				t.Errorf("%d: unexpected mutation of input: %s", i, diff.ObjectReflectDiff(original, test.in))
				continue
			}
			if !reflect.DeepEqual(out, test.out) {
				t.Errorf("%d: unexpected out: %s", i, diff.ObjectReflectDiff(out, test.out))
				continue
			}
			unsafe, err := test.scheme.UnsafeConvertToVersion(test.in, test.gv)
			if err != nil {
				t.Errorf("%d: unexpected error: %v", i, err)
				continue
			}
			if !reflect.DeepEqual(unsafe, test.out) {
				t.Errorf("%d: unexpected unsafe: %s", i, diff.ObjectReflectDiff(unsafe, test.out))
				continue
			}
			if unsafe != test.in {
				t.Errorf("%d: UnsafeConvertToVersion should return same object: %#v", i, unsafe)
				continue
			}
			continue
		}
		if !reflect.DeepEqual(out, test.out) {
			t.Errorf("%d: unexpected out: %s", i, diff.ObjectReflectDiff(out, test.out))
			continue
		}
	}
}

func TestMetaValues(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: "__internal"}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "externalVersion"}

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
	err = s.Convert(simple, external, nil)
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
