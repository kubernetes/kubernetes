/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	runtimetest "k8s.io/apimachinery/pkg/runtime/testing"
)

var (
	scheme        = runtime.NewScheme()
	codecs        = serializer.NewCodecFactory(scheme)
	cfgserializer = NewConfigSerializer(scheme, &codecs)

	intsb = runtime.NewSchemeBuilder(addInternalTypes)
	extsb = runtime.NewSchemeBuilder(registerConversions, addExternalTypes)

	groupname = "foogroup"
	intgv     = schema.GroupVersion{Group: groupname, Version: runtime.APIVersionInternal}
	extgv     = schema.GroupVersion{Group: groupname, Version: "v1alpha1"}
)

func registerConversions(s *runtime.Scheme) error {
	if err := s.AddGeneratedConversionFunc((*runtimetest.ExternalSimple)(nil), (*runtimetest.InternalSimple)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return autoConvertExternalSimpleToInternalSimple(a.(*runtimetest.ExternalSimple), b.(*runtimetest.InternalSimple), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*runtimetest.InternalSimple)(nil), (*runtimetest.ExternalSimple)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return autoConvertInternalSimpleToExternalSimple(a.(*runtimetest.InternalSimple), b.(*runtimetest.ExternalSimple), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*runtimetest.ExternalComplex)(nil), (*runtimetest.InternalComplex)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return autoConvertExternalComplexToInternalComplex(a.(*runtimetest.ExternalComplex), b.(*runtimetest.InternalComplex), scope)
	}); err != nil {
		return err
	}
	return s.AddGeneratedConversionFunc((*runtimetest.InternalComplex)(nil), (*runtimetest.ExternalComplex)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return autoConvertInternalComplexToExternalComplex(a.(*runtimetest.InternalComplex), b.(*runtimetest.ExternalComplex), scope)
	})
}

func autoConvertExternalSimpleToInternalSimple(in *runtimetest.ExternalSimple, out *runtimetest.InternalSimple, s conversion.Scope) error {
	out.TestString = in.TestString
	return nil
}

func autoConvertInternalSimpleToExternalSimple(in *runtimetest.InternalSimple, out *runtimetest.ExternalSimple, s conversion.Scope) error {
	out.TestString = in.TestString
	return nil
}

func autoConvertExternalComplexToInternalComplex(in *runtimetest.ExternalComplex, out *runtimetest.InternalComplex, s conversion.Scope) error {
	out.String = in.String
	out.Integer = in.Integer
	out.Integer64 = in.Integer64
	out.Int64 = in.Int64
	out.Bool = in.Bool
	return nil
}

func autoConvertInternalComplexToExternalComplex(in *runtimetest.InternalComplex, out *runtimetest.ExternalComplex, s conversion.Scope) error {
	out.String = in.String
	out.Integer = in.Integer
	out.Integer64 = in.Integer64
	out.Int64 = in.Int64
	out.Bool = in.Bool
	return nil
}

func addInternalTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypeWithName(intgv.WithKind("Simple"), &runtimetest.InternalSimple{})
	scheme.AddKnownTypeWithName(intgv.WithKind("Complex"), &runtimetest.InternalComplex{})
	return nil
}

func addExternalTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypeWithName(extgv.WithKind("Simple"), &runtimetest.ExternalSimple{})
	scheme.AddKnownTypeWithName(extgv.WithKind("Complex"), &runtimetest.ExternalComplex{})
	return nil
}

func init() {
	intsb.AddToScheme(scheme)
	extsb.AddToScheme(scheme)
}

var (
	oneSimple = []byte(`apiVersion: foogroup/v1alpha1
kind: Simple
testString: foo
`)
	simpleUnknownField = []byte(`apiVersion: foogroup/v1alpha1
kind: Simple
testString: foo
unknownField: bar
`)
	simpleDuplicateField = []byte(`apiVersion: foogroup/v1alpha1
kind: Simple
testString: foo
testString: bar
`)
	unrecognizedVersion = []byte(`apiVersion: foogroup/v1alpha0
kind: Simple
testString: foo
`)
	oneComplex = []byte(`Int64: 0
apiVersion: foogroup/v1alpha1
bool: false
int: 0
kind: Complex
string: bar
`)
	simpleJSON = []byte(`{"apiVersion":"foogroup/v1alpha1","kind":"Simple","testString":"foo"}
`)
	complexJSON = []byte(`{"apiVersion":"foogroup/v1alpha1","kind":"Complex","string":"bar","int":0,"Int64":0,"bool":false}
`)
)

func TestEncode(t *testing.T) {
	simpleObj := &runtimetest.InternalSimple{TestString: "foo"}
	complexObj := &runtimetest.InternalComplex{String: "bar"}
	tests := []struct {
		name        string
		format      EncodingFormat
		gv          schema.GroupVersion
		obj         runtime.Object
		expected    []byte
		expectedErr bool
	}{
		{"simple yaml", ContentTypeYAML, extgv, simpleObj, oneSimple, false},
		{"complex yaml", ContentTypeYAML, extgv, complexObj, oneComplex, false},
		{"simple json", ContentTypeJSON, extgv, simpleObj, simpleJSON, false},
		{"complex json", ContentTypeJSON, extgv, complexObj, complexJSON, false},
		{"no-conversion simple", ContentTypeJSON, extgv, &runtimetest.ExternalSimple{TestString: "foo"}, simpleJSON, false},
		{"support internal", ContentTypeJSON, intgv, simpleObj, []byte(`{"testString":"foo"}` + "\n"), false},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			actual, actualErr := cfgserializer.Encode(rt.format, rt.gv, rt.obj)
			if (actualErr != nil) != rt.expectedErr {
				t2.Errorf("expected error %t but actual %t", rt.expectedErr, actualErr != nil)
			}
			if !bytes.Equal(actual, rt.expected) {
				t2.Errorf("expected %q but actual %q", string(rt.expected), string(actual))
			}
		})
	}
}

func TestDecode(t *testing.T) {
	simpleMeta := runtime.TypeMeta{APIVersion: "foogroup/v1alpha1", Kind: "Simple"}
	complexMeta := runtime.TypeMeta{APIVersion: "foogroup/v1alpha1", Kind: "Complex"}
	tests := []struct {
		name        string
		data        []byte
		obj         runtime.Object
		expected    runtime.Object
		expectedErr bool
	}{
		{"simple internal", oneSimple, &runtimetest.InternalSimple{}, &runtimetest.InternalSimple{TestString: "foo"}, false},
		{"complex internal", oneComplex, &runtimetest.InternalComplex{}, &runtimetest.InternalComplex{String: "bar"}, false},
		{"simple external", oneSimple, &runtimetest.ExternalSimple{}, &runtimetest.ExternalSimple{TypeMeta: simpleMeta, TestString: "foo"}, false},
		{"complex external", oneComplex, &runtimetest.ExternalComplex{}, &runtimetest.ExternalComplex{TypeMeta: complexMeta, String: "bar"}, false},
		{"unknown fields", simpleUnknownField, &runtimetest.InternalSimple{}, nil, true},
		{"duplicate fields", simpleDuplicateField, &runtimetest.InternalSimple{}, nil, true},
		{"unrecognized API version", unrecognizedVersion, &runtimetest.InternalSimple{}, nil, true},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			actual := cfgserializer.DecodeInto(rt.data, rt.obj)
			if (actual != nil) != rt.expectedErr {
				t2.Errorf("expected error %t but actual %t, error:%v", rt.expectedErr, actual != nil, actual)
			}
			if rt.expected != nil && !reflect.DeepEqual(rt.obj, rt.expected) {
				t2.Errorf("expected %#v but actual %#v", rt.expected, rt.obj)
			}
		})
	}
}

func TestRoundtrip(t *testing.T) {
	tests := []struct {
		name   string
		data   []byte
		format EncodingFormat
		gv     schema.GroupVersion
		obj    runtime.Object
	}{
		{"simple yaml", oneSimple, ContentTypeYAML, extgv, &runtimetest.InternalSimple{}},
		{"complex yaml", oneComplex, ContentTypeYAML, extgv, &runtimetest.InternalComplex{}},
		{"simple json", simpleJSON, ContentTypeJSON, extgv, &runtimetest.InternalSimple{}},
		{"complex json", complexJSON, ContentTypeJSON, extgv, &runtimetest.InternalComplex{}},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			err := cfgserializer.DecodeInto(rt.data, rt.obj)
			if err != nil {
				t2.Errorf("unexpected decode error: %v", err)
			}
			actual, err := cfgserializer.Encode(rt.format, rt.gv, rt.obj)
			if err != nil {
				t2.Errorf("unexpected encode error: %v", err)
			}
			if !bytes.Equal(actual, rt.data) {
				t2.Errorf("expected %q but actual %q", string(rt.data), string(actual))
			}
		})
	}
}
