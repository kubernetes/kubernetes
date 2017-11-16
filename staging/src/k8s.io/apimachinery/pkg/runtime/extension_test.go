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
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
)

func TestEmbeddedRawExtensionMarshal(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	extension := test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}}
	data, err := json.Marshal(extension)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(data) != `{"Ext":{"foo":"bar"}}` {
		t.Errorf("unexpected data: %s", string(data))
	}
}
func TestEmbeddedRawExtensionUnmarshal(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	testCases := map[string]struct {
		orig test
	}{
		"non-empty object": {
			orig: test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}},
		},
		"empty object": {
			orig: test{Ext: runtime.RawExtension{}},
		},
	}

	for k, tc := range testCases {
		new := test{}
		data, _ := json.Marshal(tc.orig)
		if err := json.Unmarshal(data, &new); err != nil {
			t.Errorf("%s: umarshal error: %v", k, err)
		}
		if !reflect.DeepEqual(tc.orig, new) {
			t.Errorf("%s: unmarshaled struct differs from original: %v %v", k, tc.orig, new)
		}
	}
}

func TestEmbeddedRawExtensionRoundTrip(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	testCases := map[string]struct {
		orig test
	}{
		"non-empty object": {
			orig: test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}},
		},
		"empty object": {
			orig: test{Ext: runtime.RawExtension{}},
		},
	}

	for k, tc := range testCases {
		new1 := test{}
		new2 := test{}
		data, err := json.Marshal(tc.orig)
		if err != nil {
			t.Errorf("1st marshal error: %v", err)
		}
		if err = json.Unmarshal(data, &new1); err != nil {
			t.Errorf("1st unmarshal error: %v", err)
		}
		newData, err := json.Marshal(new1)
		if err != nil {
			t.Errorf("2st marshal error: %v", err)
		}
		if err = json.Unmarshal(newData, &new2); err != nil {
			t.Errorf("2nd unmarshal error: %v", err)
		}
		if !bytes.Equal(data, newData) {
			t.Errorf("%s: re-marshaled data differs from original: %v %v", k, data, newData)
		}
		if !reflect.DeepEqual(tc.orig, new1) {
			t.Errorf("%s: unmarshaled struct differs from original: %v %v", k, tc.orig, new1)
		}
		if !reflect.DeepEqual(new1, new2) {
			t.Errorf("%s: re-unmarshaled struct differs from original: %v %v", k, new1, new2)
		}
	}
}
