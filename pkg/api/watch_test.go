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

func TestEmbeddedDefaultSerialization(t *testing.T) {
	expected := WatchEvent{
		Type:   "foo",
		Object: EmbeddedObject{&Pod{}},
	}
	data, err := json.Marshal(expected)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	actual := WatchEvent{}
	if err := json.Unmarshal(data, &actual); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}
