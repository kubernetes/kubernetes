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

package runtime_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

var scheme = runtime.NewScheme()
var Codec = runtime.CodecFor(scheme, "v1test")

type EmbeddedTest struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`
	ID               string                 `yaml:"id,omitempty" json:"id,omitempty"`
	Object           runtime.EmbeddedObject `yaml:"object,omitempty" json:"object,omitempty"`
	EmptyObject      runtime.EmbeddedObject `yaml:"emptyObject,omitempty" json:"emptyObject,omitempty"`
}

type EmbeddedTestExternal struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`
	ID               string               `yaml:"id,omitempty" json:"id,omitempty"`
	Object           runtime.RawExtension `yaml:"object,omitempty" json:"object,omitempty"`
	EmptyObject      runtime.RawExtension `yaml:"emptyObject,omitempty" json:"emptyObject,omitempty"`
}

func (*EmbeddedTest) IsAnAPIObject()         {}
func (*EmbeddedTestExternal) IsAnAPIObject() {}

func TestEmbeddedObject(t *testing.T) {
	s := scheme
	s.AddKnownTypes("", &EmbeddedTest{})
	s.AddKnownTypeWithName("v1test", "EmbeddedTest", &EmbeddedTestExternal{})

	outer := &EmbeddedTest{
		TypeMeta: runtime.TypeMeta{Name: "outer"},
		ID:       "outer",
		Object: runtime.EmbeddedObject{
			&EmbeddedTest{
				TypeMeta: runtime.TypeMeta{Name: "inner"},
				ID:       "inner",
			},
		},
	}

	wire, err := s.EncodeToVersion(outer, "v1test")
	if err != nil {
		t.Fatalf("Unexpected encode error '%v'", err)
	}

	t.Logf("Wire format is:\n%v\n", string(wire))

	decoded, err := s.Decode(wire)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}

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
	if externalViaJSON.Kind == "" || externalViaJSON.APIVersion == "" || externalViaJSON.Name != "outer" {
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
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}
	if a := decodedViaJSON; a.Object.Object != nil || a.EmptyObject.Object != nil {
		t.Errorf("Expected embedded objects to be nil: %#v", a)
	}
}
