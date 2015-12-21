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

package json

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestSimpleMetaFactoryInterpret(t *testing.T) {
	factory := SimpleMetaFactory{}
	gvk, err := factory.Interpret([]byte(`{"apiVersion":"1","kind":"object"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gvk.Version != "1" || gvk.Kind != "object" {
		t.Errorf("unexpected interpret: %#v", gvk)
	}

	// no kind or version
	gvk, err = factory.Interpret([]byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gvk.Version != "" || gvk.Kind != "" {
		t.Errorf("unexpected interpret: %#v", gvk)
	}

	// unparsable
	gvk, err = factory.Interpret([]byte(`{`))
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestMetaValues(t *testing.T) {
	type InternalSimple struct {
		APIVersion string `json:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty"`
		TestString string `json:"testString"`
	}
	type ExternalSimple struct {
		APIVersion string `json:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty"`
		TestString string `json:"testString"`
	}
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "externalVersion"}

	s := conversion.NewScheme()
	s.AddKnownTypeWithName(internalGV.WithKind("Simple"), &InternalSimple{})
	s.AddKnownTypeWithName(externalGV.WithKind("Simple"), &ExternalSimple{})

	internalToExternalCalls := 0
	externalToInternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			t.Logf("internal -> external")
			if e, a := internalGV.String(), scope.Meta().SrcVersion; e != a {
				t.Fatalf("Expected '%v', got '%v'", e, a)
			}
			if e, a := externalGV.String(), scope.Meta().DestVersion; e != a {
				t.Fatalf("Expected '%v', got '%v'", e, a)
			}
			scope.Convert(&in.TestString, &out.TestString, 0)
			internalToExternalCalls++
			return nil
		},
		func(in *ExternalSimple, out *InternalSimple, scope conversion.Scope) error {
			t.Logf("external -> internal")
			if e, a := externalGV.String(), scope.Meta().SrcVersion; e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			if e, a := internalGV.String(), scope.Meta().DestVersion; e != a {
				t.Fatalf("Expected '%v', got '%v'", e, a)
			}
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

	out, err := s.ConvertToVersion(simple, externalGV.String())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	internal, err := s.ConvertToVersion(out, internalGV.String())
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
	s := conversion.NewScheme()
	// We deliberately don't register the types.

	internalToExternalCalls := 0

	// Register functions to verify that scope.Meta() gets set correctly.
	err := s.AddConversionFuncs(
		func(in *InternalSimple, out *ExternalSimple, scope conversion.Scope) error {
			if e, a := "unknown/unknown", scope.Meta().SrcVersion; e != a {
				t.Fatalf("Expected '%v', got '%v'", e, a)
			}
			if e, a := "unknown/unknown", scope.Meta().DestVersion; e != a {
				t.Fatalf("Expected '%v', got '%v'", e, a)
			}
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

type testMetaFactory struct{}

func (testMetaFactory) Interpret(data []byte) (*unversioned.GroupVersionKind, error) {
	findKind := struct {
		APIVersion string `json:"myVersionKey,omitempty"`
		ObjectKind string `json:"myKindKey,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	if err := yaml.Unmarshal(data, &findKind); err != nil {
		return nil, fmt.Errorf("couldn't get version/kind: %v", err)
	}
	gv, err := unversioned.ParseGroupVersion(findKind.APIVersion)
	if err != nil {
		return nil, err
	}
	return &unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: findKind.ObjectKind}, nil
}
