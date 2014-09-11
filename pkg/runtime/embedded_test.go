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

// EmbeddedObject implements a Codec specific version of an
// embedded object.
type EmbeddedObject struct {
	runtime.Object
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (a *EmbeddedObject) UnmarshalJSON(b []byte) error {
	obj, err := runtime.CodecUnmarshalJSON(Codec, b)
	a.Object = obj
	return err
}

// MarshalJSON implements the json.Marshaler interface.
func (a EmbeddedObject) MarshalJSON() ([]byte, error) {
	return runtime.CodecMarshalJSON(Codec, a.Object)
}

// SetYAML implements the yaml.Setter interface.
func (a *EmbeddedObject) SetYAML(tag string, value interface{}) bool {
	obj, ok := runtime.CodecSetYAML(Codec, tag, value)
	a.Object = obj
	return ok
}

// GetYAML implements the yaml.Getter interface.
func (a EmbeddedObject) GetYAML() (tag string, value interface{}) {
	return runtime.CodecGetYAML(Codec, a.Object)
}

type EmbeddedTest struct {
	runtime.JSONBase `yaml:",inline" json:",inline"`
	Object           EmbeddedObject `yaml:"object,omitempty" json:"object,omitempty"`
	EmptyObject      EmbeddedObject `yaml:"emptyObject,omitempty" json:"emptyObject,omitempty"`
}

func (*EmbeddedTest) IsAnAPIObject() {}

func TestEmbeddedObject(t *testing.T) {
	s := scheme
	s.AddKnownTypes("", &EmbeddedTest{})
	s.AddKnownTypes("v1test", &EmbeddedTest{})

	outer := &EmbeddedTest{
		JSONBase: runtime.JSONBase{ID: "outer"},
		Object: EmbeddedObject{
			&EmbeddedTest{
				JSONBase: runtime.JSONBase{ID: "inner"},
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

	// test JSON decoding, too, since Decode uses yaml unmarshalling.
	var decodedViaJSON EmbeddedTest
	err = json.Unmarshal(wire, &decodedViaJSON)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}

	// Things that Decode would have done for us:
	decodedViaJSON.Kind = ""
	decodedViaJSON.APIVersion = ""

	if e, a := outer, &decodedViaJSON; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v but got %#v", e, a)
	}
}
