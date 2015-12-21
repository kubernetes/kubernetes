/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package registered

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestAllPreferredGroupVersions(t *testing.T) {
	testCases := []struct {
		groupMetas []apimachinery.GroupMeta
		expect     string
	}{
		{
			groupMetas: []apimachinery.GroupMeta{
				{
					GroupVersion: unversioned.GroupVersion{"group1", "v1"},
				},
				{
					GroupVersion: unversioned.GroupVersion{"group2", "v2"},
				},
				{
					GroupVersion: unversioned.GroupVersion{"", "v1"},
				},
			},
			expect: "group1/v1,group2/v2,v1",
		},
		{
			groupMetas: []apimachinery.GroupMeta{
				{
					GroupVersion: unversioned.GroupVersion{"", "v1"},
				},
			},
			expect: "v1",
		},
		{
			groupMetas: []apimachinery.GroupMeta{},
			expect:     "",
		},
	}
	for _, testCase := range testCases {
		for _, groupMeta := range testCase.groupMetas {
			RegisterGroup(groupMeta)
		}
		output := AllPreferredGroupVersions()
		if testCase.expect != output {
			t.Errorf("Error. expect: %s, got: %s", testCase.expect, output)
		}
		reset()
	}
}

var status = &unversioned.Status{
	Status:  unversioned.StatusFailure,
	Code:    200,
	Reason:  unversioned.StatusReasonUnknown,
	Message: "",
}

func TestV1EncodeDecodeStatus(t *testing.T) {
	v1Codec := testapi.Default.Codec()

	encoded, err := runtime.Encode(v1Codec, status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %v", string(encoded))
	}
	if typeMeta.APIVersion != "v1" {
		t.Errorf("APIVersion is not set to \"v1\". Got %v", string(encoded))
	}
	decoded, err := runtime.Decode(v1Codec, encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %#v, got: %#v", status, decoded)
	}
}

func TestExperimentalEncodeDecodeStatus(t *testing.T) {
	extensionCodec := testapi.Extensions.Codec()
	encoded, err := runtime.Encode(extensionCodec, status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %s", encoded)
	}
	if typeMeta.APIVersion != "v1" {
		t.Errorf("APIVersion is not set to \"\". Got %s", encoded)
	}
	decoded, err := runtime.Decode(extensionCodec, encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %v, got: %v", status, decoded)
	}
}
