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

package api

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestAPIObject(t *testing.T) {
	type EmbeddedTest struct {
		JSONBase    `yaml:",inline" json:",inline"`
		Object      APIObject `yaml:"object,omitempty" json:"object,omitempty"`
		EmptyObject APIObject `yaml:"emptyObject,omitempty" json:"emptyObject,omitempty"`
	}
	AddKnownTypes("", EmbeddedTest{})
	AddKnownTypes("v1beta1", EmbeddedTest{})

	outer := &EmbeddedTest{
		JSONBase: JSONBase{ID: "outer"},
		Object: APIObject{
			&EmbeddedTest{
				JSONBase: JSONBase{ID: "inner"},
			},
		},
	}

	wire, err := Encode(outer)
	if err != nil {
		t.Fatalf("Unexpected encode error '%v'", err)
	}

	t.Logf("Wire format is:\n%v\n", string(wire))

	decoded, err := Decode(wire)
	if err != nil {
		t.Fatalf("Unexpected decode error %v", err)
	}

	if e, a := outer, decoded; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v but got %#v", e, a)
	}

	// test JSON decoding, too, since api.Decode uses yaml unmarshalling.
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
