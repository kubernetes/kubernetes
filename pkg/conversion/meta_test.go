/*
Copyright 2014 Google Inc. All rights reserved.

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

package conversion

import (
	"reflect"
	"testing"
)

func TestMetaValues(t *testing.T) {
	type InternalSimple struct {
		Version    string `json:"version,omitempty" yaml:"version,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		TestString string `json:"testString" yaml:"testString"`
	}
	type ExternalSimple struct {
		Version    string `json:"version,omitempty" yaml:"version,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		TestString string `json:"testString" yaml:"testString"`
	}
	s := NewScheme()
	s.AddKnownTypeWithName("", "Simple", &InternalSimple{})
	s.AddKnownTypeWithName("externalVersion", "Simple", &ExternalSimple{})

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope Scope) error {
			if e, a := "", scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := "externalVersion", scope.Meta().DestVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *ExternalSimple, out *InternalSimple, scope Scope) error {
			if e, a := "externalVersion", scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := "", scope.Meta().DestVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TestString, &out.TestString, 0)
			externalToInternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	simple := &InternalSimple{
		TestString: "foo",
	}

	// Test Encode, Decode, and DecodeInto
	data, err := s.EncodeToVersion(simple, "externalVersion")
	obj2, err2 := s.Decode(data)
	obj3 := &InternalSimple{}
	err3 := s.DecodeInto(data, obj3)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v' '%v'", err, err2, err3)
	}
	if _, ok := obj2.(*InternalSimple); !ok {
		t.Fatalf("Got wrong type")
	}
	if e, a := simple, obj2; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}
	if e, a := simple, obj3; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", e, a)
	}

	// Test Convert
	external := &ExternalSimple{}
	err = s.Convert(simple, external)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, external.TestString; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Encode and Convert should each have caused an increment.
	if e, a := 2, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	// Decode and DecodeInto should each have caused an increment.
	if e, a := 2, externalToInternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestMetaValuesUnregisteredConvert(t *testing.T) {
	type InternalSimple struct {
		Version    string `json:"version,omitempty" yaml:"version,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		TestString string `json:"testString" yaml:"testString"`
	}
	type ExternalSimple struct {
		Version    string `json:"version,omitempty" yaml:"version,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		TestString string `json:"testString" yaml:"testString"`
	}
	s := NewScheme()
	s.InternalVersion = ""
	// We deliberately don't register the types.

	internalToExternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope Scope) error {
			if e, a := "unknown", scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := "unknown", scope.Meta().DestVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
	)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	simple := &InternalSimple{TestString: "foo"}
	external := &ExternalSimple{}
	err = s.Convert(simple, external)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := simple.TestString, external.TestString; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Verify that our conversion handler got called.
	if e, a := 1, internalToExternalCalls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}
