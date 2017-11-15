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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/diff"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

func TestScheme(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &runtimetesting.InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &runtimetesting.ExternalSimple{})

	// If set, would clear TypeMeta during conversion.
	//scheme.AddIgnoredConversionType(&TypeMeta{}, &TypeMeta{})

	// test that scheme is an ObjectTyper
	var _ runtime.ObjectTyper = scheme

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := scheme.AddConversionFuncs(
		func(in *runtimetesting.InternalSimple, out *runtimetesting.ExternalSimple, scope conversion.Scope) error {
			scope.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *runtimetesting.ExternalSimple, out *runtimetesting.InternalSimple, scope conversion.Scope) error {
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

	simple := &runtimetesting.InternalSimple{
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
	if _, ok := obj2.(*runtimetesting.InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if e, a := simple, obj2; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}

	obj3 := &runtimetesting.InternalSimple{}
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
	if _, ok := obj4.(*runtimetesting.ExternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}

	// Test Convert
	external := &runtimetesting.ExternalSimple{}
	err = scheme.Convert(simple, external, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, external.TestString; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Test convert internal to unstructured
	unstructuredObj := &runtimetesting.Unstructured{}
	err = scheme.Convert(simple, unstructuredObj, nil)
	if err == nil || !strings.Contains(err.Error(), "to Unstructured without providing a preferred version to convert to") {
		t.Fatalf("Unexpected non-error: %v", err)
	}
	err = scheme.Convert(simple, unstructuredObj, schema.GroupVersion{Group: "test.group", Version: "testExternal"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, unstructuredObj.Object["testString"].(string); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e := unstructuredObj.GetObjectKind().GroupVersionKind(); !reflect.DeepEqual(e, schema.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "Simple"}) {
		t.Errorf("Unexpected object kind: %#v", e)
	}
	if gvks, unversioned, err := scheme.ObjectKinds(unstructuredObj); err != nil || !reflect.DeepEqual(gvks[0], schema.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "Simple"}) || unversioned {
		t.Errorf("Scheme did not recognize unversioned: %v, %#v %t", err, gvks, unversioned)
	}

	// Test convert external to unstructured
	unstructuredObj = &runtimetesting.Unstructured{}
	err = scheme.Convert(external, unstructuredObj, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, unstructuredObj.Object["testString"].(string); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e := unstructuredObj.GetObjectKind().GroupVersionKind(); !reflect.DeepEqual(e, schema.GroupVersionKind{Group: "test.group", Version: "testExternal", Kind: "Simple"}) {
		t.Errorf("Unexpected object kind: %#v", e)
	}

	// Test convert unstructured to unstructured
	uIn := &runtimetesting.Unstructured{Object: map[string]interface{}{
		"test": []interface{}{"other", "test"},
	}}
	uOut := &runtimetesting.Unstructured{}
	err = scheme.Convert(uIn, uOut, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(uIn.Object, uOut.Object) {
		t.Errorf("Unexpected object contents: %#v", uOut.Object)
	}

	// Test convert unstructured to structured
	externalOut := &runtimetesting.ExternalSimple{}
	err = scheme.Convert(unstructuredObj, externalOut, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(external, externalOut) {
		t.Errorf("Unexpected object contents: %#v", externalOut)
	}

	// Encode and Convert should each have caused an increment.
	if e, a := 3, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	// DecodeInto and Decode should each have caused an increment because of a conversion
	if e, a := 2, externalToInternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Verify that unstructured types must have V and K set
	emptyObj := &runtimetesting.Unstructured{Object: make(map[string]interface{})}
	if _, _, err := scheme.ObjectKinds(emptyObj); !runtime.IsMissingKind(err) {
		t.Errorf("unexpected error: %v", err)
	}
	emptyObj.SetGroupVersionKind(schema.GroupVersionKind{Kind: "Test"})
	if _, _, err := scheme.ObjectKinds(emptyObj); !runtime.IsMissingVersion(err) {
		t.Errorf("unexpected error: %v", err)
	}
	emptyObj.SetGroupVersionKind(schema.GroupVersionKind{Kind: "Test", Version: "v1"})
	if _, _, err := scheme.ObjectKinds(emptyObj); err != nil {
		t.Errorf("unexpected error: %v", err)
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

func TestExternalToInternalMapping(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "testExternal"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("OptionalExtensionType"), &runtimetesting.InternalOptionalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("OptionalExtensionType"), &runtimetesting.ExternalOptionalExtensionType{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	table := []struct {
		obj     runtime.Object
		encoded string
	}{
		{
			&runtimetesting.InternalOptionalExtensionType{Extension: nil},
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
	scheme.AddKnownTypeWithName(internalGV.WithKind("ExtensionType"), &runtimetesting.InternalExtensionType{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("OptionalExtensionType"), &runtimetesting.InternalOptionalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("ExtensionType"), &runtimetesting.ExternalExtensionType{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("OptionalExtensionType"), &runtimetesting.ExternalOptionalExtensionType{})

	// register external first when the object is the same in both schemes, so ObjectVersionAndKind reports the
	// external version.
	scheme.AddKnownTypeWithName(externalGV.WithKind("A"), &runtimetesting.ExtensionA{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("B"), &runtimetesting.ExtensionB{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("A"), &runtimetesting.ExtensionA{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("B"), &runtimetesting.ExtensionB{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	table := []struct {
		obj      runtime.Object
		expected runtime.Object
		encoded  string
	}{
		{
			&runtimetesting.InternalExtensionType{
				Extension: runtime.NewEncodable(codec, &runtimetesting.ExtensionA{TestString: "foo"}),
			},
			&runtimetesting.InternalExtensionType{
				Extension: &runtime.Unknown{
					Raw:         []byte(`{"apiVersion":"test.group/testExternal","kind":"A","testString":"foo"}`),
					ContentType: runtime.ContentTypeJSON,
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"apiVersion":"` + externalGV.String() + `","kind":"ExtensionType","extension":{"apiVersion":"test.group/testExternal","kind":"A","testString":"foo"}}
`,
		}, {
			&runtimetesting.InternalExtensionType{Extension: runtime.NewEncodable(codec, &runtimetesting.ExtensionB{TestString: "bar"})},
			&runtimetesting.InternalExtensionType{
				Extension: &runtime.Unknown{
					Raw:         []byte(`{"apiVersion":"test.group/testExternal","kind":"B","testString":"bar"}`),
					ContentType: runtime.ContentTypeJSON,
				},
			},
			// apiVersion is set in the serialized object for easier consumption by clients
			`{"apiVersion":"` + externalGV.String() + `","kind":"ExtensionType","extension":{"apiVersion":"test.group/testExternal","kind":"B","testString":"bar"}}
`,
		}, {
			&runtimetesting.InternalExtensionType{Extension: nil},
			&runtimetesting.InternalExtensionType{
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
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &runtimetesting.InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &runtimetesting.ExternalSimple{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	test := &runtimetesting.InternalSimple{
		TestString: "I'm the same",
	}
	obj := runtime.Object(test)
	data, err := runtime.Encode(codec, obj)
	obj2, gvk, err2 := codec.Decode(data, nil, nil)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*runtimetesting.InternalSimple); !ok {
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
	scheme.AddUnversionedTypes(externalGV, &runtimetesting.InternalSimple{})
	scheme.AddKnownTypeWithName(internalGV.WithKind("Simple"), &runtimetesting.InternalSimple{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Simple"), &runtimetesting.ExternalSimple{})
	scheme.AddKnownTypeWithName(otherGV.WithKind("Simple"), &runtimetesting.ExternalSimple{})

	codec := serializer.NewCodecFactory(scheme).LegacyCodec(externalGV)

	if unv, ok := scheme.IsUnversioned(&runtimetesting.InternalSimple{}); !unv || !ok {
		t.Fatalf("type not unversioned and in scheme: %t %t", unv, ok)
	}

	kinds, _, err := scheme.ObjectKinds(&runtimetesting.InternalSimple{})
	if err != nil {
		t.Fatal(err)
	}
	kind := kinds[0]
	if kind != externalGV.WithKind("InternalSimple") {
		t.Fatalf("unexpected: %#v", kind)
	}

	test := &runtimetesting.InternalSimple{
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
	if _, ok := obj2.(*runtimetesting.InternalSimple); !ok {
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

// TestObjectFuzzer can randomly populate all the above objects.
var TestObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 100).Funcs(
	func(j *runtimetesting.MyWeirdCustomEmbeddedVersionKindField, c fuzz.Continue) {
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
	alternateInternalGV := schema.GroupVersion{Group: "custom", Version: "__internal"}
	differentExternalGV := schema.GroupVersion{Group: "other", Version: "v2"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &runtimetesting.TestType1{}, &runtimetesting.TestType2{}, &runtimetesting.ExternalInternalSame{})
	s.AddKnownTypes(externalGV, &runtimetesting.ExternalInternalSame{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType2"), &runtimetesting.ExternalTestType2{})
	s.AddKnownTypeWithName(internalGV.WithKind("TestType3"), &runtimetesting.TestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType3"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType4"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(alternateInternalGV.WithKind("TestType3"), &runtimetesting.TestType1{})
	s.AddKnownTypeWithName(alternateExternalGV.WithKind("TestType3"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(alternateExternalGV.WithKind("TestType5"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(differentExternalGV.WithKind("TestType1"), &runtimetesting.ExternalTestType1{})
	s.AddUnversionedTypes(externalGV, &runtimetesting.UnversionedType{})

	s.AddConversionFuncs(func(in *runtimetesting.TestType1, out *runtimetesting.ExternalTestType1, s conversion.Scope) {
		out.A = in.A
	})
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

func TestAddKnownTypesIdemPotent(t *testing.T) {
	s := runtime.NewScheme()

	gv := schema.GroupVersion{Group: "foo", Version: "v1"}
	s.AddKnownTypes(gv, &runtimetesting.InternalSimple{})
	s.AddKnownTypes(gv, &runtimetesting.InternalSimple{})
	if len(s.KnownTypes(gv)) != 1 {
		t.Errorf("expected only one %v type after double registration", gv)
	}
	if len(s.AllKnownTypes()) != 1 {
		t.Errorf("expected only one type after double registration")
	}

	s.AddKnownTypeWithName(gv.WithKind("InternalSimple"), &runtimetesting.InternalSimple{})
	s.AddKnownTypeWithName(gv.WithKind("InternalSimple"), &runtimetesting.InternalSimple{})
	if len(s.KnownTypes(gv)) != 1 {
		t.Errorf("expected only one %v type after double registration with custom name", gv)
	}
	if len(s.AllKnownTypes()) != 1 {
		t.Errorf("expected only one type after double registration with custom name")
	}

	s.AddUnversionedTypes(gv, &runtimetesting.InternalSimple{})
	s.AddUnversionedTypes(gv, &runtimetesting.InternalSimple{})
	if len(s.KnownTypes(gv)) != 1 {
		t.Errorf("expected only one %v type after double registration with custom name", gv)
	}
	if len(s.AllKnownTypes()) != 1 {
		t.Errorf("expected only one type after double registration with custom name")
	}

	kinds, _, err := s.ObjectKinds(&runtimetesting.InternalSimple{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(kinds) != 1 {
		t.Errorf("expected only one kind for InternalSimple after double registration")
	}
}

// redefine InternalSimple with the same name, but obviously as a different type than in runtimetesting
type InternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

func (s *InternalSimple) DeepCopyObject() runtime.Object { return nil }

func TestConflictingAddKnownTypes(t *testing.T) {
	s := runtime.NewScheme()
	gv := schema.GroupVersion{Group: "foo", Version: "v1"}

	panicked := make(chan bool)
	go func() {
		defer func() {
			if recover() != nil {
				panicked <- true
			}
		}()
		s.AddKnownTypeWithName(gv.WithKind("InternalSimple"), &runtimetesting.InternalSimple{})
		s.AddKnownTypeWithName(gv.WithKind("InternalSimple"), &runtimetesting.ExternalSimple{})
		panicked <- false
	}()
	if !<-panicked {
		t.Errorf("Expected AddKnownTypesWithName to panic with conflicting type registrations")
	}

	go func() {
		defer func() {
			if recover() != nil {
				panicked <- true
			}
		}()

		s.AddUnversionedTypes(gv, &runtimetesting.InternalSimple{})
		s.AddUnversionedTypes(gv, &InternalSimple{})
		panicked <- false
	}()
	if !<-panicked {
		t.Errorf("Expected AddUnversionedTypes to panic with conflicting type registrations")
	}
}

func TestConvertToVersionBasic(t *testing.T) {
	s := GetTestScheme()
	tt := &runtimetesting.TestType1{A: "I'm not a pointer object"}
	other, err := s.ConvertToVersion(tt, schema.GroupVersion{Version: "v1"})
	if err != nil {
		t.Fatalf("Failure: %v", err)
	}
	converted, ok := other.(*runtimetesting.ExternalTestType1)
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
			in:     &runtimetesting.UnknownType{},
			errFn:  func(err error) bool { return err != nil && runtime.IsNotRegisteredError(err) },
		},
		// errors if the group versioner returns no target
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{},
			errFn: func(err error) bool {
				return err != nil && strings.Contains(err.Error(), "is not suitable for converting")
			},
		},
		// converts to internal
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     schema.GroupVersion{Version: "__internal"},
			out:    &runtimetesting.TestType1{A: "test"},
		},
		// converts from unstructured to internal
		{
			scheme: GetTestScheme(),
			in: &runtimetesting.Unstructured{Object: map[string]interface{}{
				"apiVersion": "custom/v1",
				"kind":       "TestType3",
				"A":          "test",
			}},
			gv:  schema.GroupVersion{Version: "__internal"},
			out: &runtimetesting.TestType1{A: "test"},
		},
		// converts from unstructured to external
		{
			scheme: GetTestScheme(),
			in: &runtimetesting.Unstructured{Object: map[string]interface{}{
				"apiVersion": "custom/v1",
				"kind":       "TestType3",
				"A":          "test",
			}},
			gv:  schema.GroupVersion{Group: "custom", Version: "v1"},
			out: &runtimetesting.ExternalTestType1{MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType3"}, A: "test"},
		},
		// prefers the best match
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "__internal"}, {Version: "v1"}},
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// unversioned type returned as-is
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.UnversionedType{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}},
			same:   true,
			out: &runtimetesting.UnversionedType{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "UnversionedType"},
				A: "test",
			},
		},
		// unversioned type returned when not included in the target types
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.UnversionedType{A: "test"},
			gv:     schema.GroupVersions{{Group: "other", Version: "v2"}},
			same:   true,
			out: &runtimetesting.UnversionedType{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "UnversionedType"},
				A: "test",
			},
		},
		// detected as already being in the target version
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// detected as already being in the first target version
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}, {Version: "__internal"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// detected as already being in the first target version
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     schema.GroupVersions{{Version: "v1"}, {Version: "__internal"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (1/3): different kind
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType3", Version: "v1"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (2/3): different gv
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType3", Group: "custom", Version: "v1"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// the external type is registered in multiple groups, versions, and kinds, and can be targeted to all of them (3/3): different gvk
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Group: "custom", Version: "v1", Kind: "TestType5"}},
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType5"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, copies because version differs
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "other", Version: "v2"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}, schema.GroupKind{Kind: "TestType1"}),
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "other/v2", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, copies because version differs
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "other", Version: "v2"}, schema.GroupKind{Kind: "TestType1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}),
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "other/v2", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner is unable to find a match when kind AND group don't match (there is no TestType1 kind in group "other", and no kind "TestType5" in the default group)
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.TestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "custom", Version: "v1"}, schema.GroupKind{Group: "other"}, schema.GroupKind{Kind: "TestType5"}),
			errFn: func(err error) bool {
				return err != nil && strings.Contains(err.Error(), "is not suitable for converting")
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, performs no copy
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "", Version: "v1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}, schema.GroupKind{Kind: "TestType1"}),
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// multi group versioner recognizes multiple groups and forces the output to a particular version, performs no copy
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.ExternalTestType1{A: "test"},
			gv:     runtime.NewMultiGroupVersioner(schema.GroupVersion{Group: "", Version: "v1"}, schema.GroupKind{Kind: "TestType1"}, schema.GroupKind{Group: "custom", Kind: "TestType3"}),
			same:   true,
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType1"},
				A: "test",
			},
		},
		// group versioner can choose a particular target kind for a given input when kind is the same across group versions
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.TestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Version: "v1", Kind: "TestType3"}},
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "v1", ObjectKind: "TestType3"},
				A: "test",
			},
		},
		// group versioner can choose a different kind
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.TestType1{A: "test"},
			gv:     testGroupVersioner{ok: true, target: schema.GroupVersionKind{Kind: "TestType5", Group: "custom", Version: "v1"}},
			out: &runtimetesting.ExternalTestType1{
				MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{APIVersion: "custom/v1", ObjectKind: "TestType5"},
				A: "test",
			},
		},
	}
	for i, test := range testCases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			original := test.in.DeepCopyObject()
			out, err := test.scheme.ConvertToVersion(test.in, test.gv)
			switch {
			case test.errFn != nil:
				if !test.errFn(err) {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			case err != nil:
				t.Fatalf("unexpected error: %v", err)
			}
			if out == test.in {
				t.Fatalf("ConvertToVersion should always copy out: %#v", out)
			}

			if test.same {
				if !reflect.DeepEqual(original, test.in) {
					t.Fatalf("unexpected mutation of input: %s", diff.ObjectReflectDiff(original, test.in))
				}
				if !reflect.DeepEqual(out, test.out) {
					t.Fatalf("unexpected out: %s", diff.ObjectReflectDiff(out, test.out))
				}
				unsafe, err := test.scheme.UnsafeConvertToVersion(test.in, test.gv)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if !reflect.DeepEqual(unsafe, test.out) {
					t.Fatalf("unexpected unsafe: %s", diff.ObjectReflectDiff(unsafe, test.out))
				}
				if unsafe != test.in {
					t.Fatalf("UnsafeConvertToVersion should return same object: %#v", unsafe)
				}
				return
			}
			if !reflect.DeepEqual(out, test.out) {
				t.Fatalf("unexpected out: %s", diff.ObjectReflectDiff(out, test.out))
			}
		})
	}
}

func TestConvert(t *testing.T) {
	testCases := []struct {
		scheme *runtime.Scheme
		in     runtime.Object
		into   runtime.Object
		gv     runtime.GroupVersioner
		out    runtime.Object
		errFn  func(error) bool
	}{
		// converts from internal to unstructured, given a target version
		{
			scheme: GetTestScheme(),
			in:     &runtimetesting.TestType1{A: "test"},
			into:   &runtimetesting.Unstructured{},
			out: &runtimetesting.Unstructured{Object: map[string]interface{}{
				"myVersionKey": "custom/v1",
				"myKindKey":    "TestType3",
				"A":            "test",
			}},
			gv: schema.GroupVersion{Group: "custom", Version: "v1"},
		},
	}
	for i, test := range testCases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			err := test.scheme.Convert(test.in, test.into, test.gv)
			switch {
			case test.errFn != nil:
				if !test.errFn(err) {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			case err != nil:
				t.Fatalf("unexpected error: %v", err)
				return
			}

			if !reflect.DeepEqual(test.into, test.out) {
				t.Fatalf("unexpected out: %s", diff.ObjectReflectDiff(test.into, test.out))
			}
		})
	}
}

func TestMetaValues(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: "__internal"}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "externalVersion"}

	s := runtime.NewScheme()
	s.AddKnownTypeWithName(internalGV.WithKind("Simple"), &runtimetesting.InternalSimple{})
	s.AddKnownTypeWithName(externalGV.WithKind("Simple"), &runtimetesting.ExternalSimple{})

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *runtimetesting.InternalSimple, out *runtimetesting.ExternalSimple, scope conversion.Scope) error {
			t.Logf("internal -> external")
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *runtimetesting.ExternalSimple, out *runtimetesting.InternalSimple, scope conversion.Scope) error {
			t.Logf("external -> internal")
			scope.Convert(&in.TestString, &out.TestString, 0)
			externalToInternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	simple := &runtimetesting.InternalSimple{
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
