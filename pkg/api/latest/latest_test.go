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

package latest_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestAllPreferredGroupVersions(t *testing.T) {
	testCases := []struct {
		groupMetaMap latest.GroupMetaMap
		expect       string
	}{
		{
			groupMetaMap: latest.GroupMetaMap{
				"group1": &latest.GroupMeta{
					GroupVersion: unversioned.GroupVersion{"group1", "v1"},
				},
				"group2": &latest.GroupMeta{
					GroupVersion: unversioned.GroupVersion{"group2", "v2"},
				},
				"": &latest.GroupMeta{
					GroupVersion: unversioned.GroupVersion{"", "v1"},
				},
			},
			expect: "group1/v1,group2/v2,v1",
		},
		{
			groupMetaMap: latest.GroupMetaMap{
				"": &latest.GroupMeta{
					GroupVersion: unversioned.GroupVersion{"", "v1"},
				},
			},
			expect: "v1",
		},
		{
			groupMetaMap: latest.GroupMetaMap{},
			expect:       "",
		},
	}
	for _, testCase := range testCases {
		output := testCase.groupMetaMap.AllPreferredGroupVersions()
		if testCase.expect != output {
			t.Errorf("Error. expect: %s, got: %s", testCase.expect, output)
		}
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
	// TODO: caesarxuchao: use the testapi.Extensions.Codec() once the PR that
	// moves experimental from v1 to v1beta1 got merged.
	expCodec := testapi.Extensions.Codec()
	//expCodec := runtime.CodecFor(api.Scheme, "extensions/v1beta1")
	encoded, err := runtime.Encode(expCodec, status)
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
	decoded, err := runtime.Decode(expCodec, encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %v, got: %v", status, decoded)
	}
}

func TestUnversioned(t *testing.T) {
	for _, obj := range []runtime.Object{
		&unversioned.Status{},
		&unversioned.ExportOptions{},
	} {
		if unversioned, ok := api.Scheme.IsUnversioned(obj); !unversioned || !ok {
			t.Errorf("%v is expected to be unversioned", reflect.TypeOf(obj))
		}
	}
}
