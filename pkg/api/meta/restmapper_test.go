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

package meta

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type fakeCodec struct{}

func (fakeCodec) Encode(runtime.Object) ([]byte, error) {
	return []byte{}, nil
}

func (fakeCodec) Decode([]byte) (runtime.Object, error) {
	return nil, nil
}

func (fakeCodec) DecodeInto([]byte, runtime.Object) error {
	return nil
}

type fakeConvertor struct{}

func (fakeConvertor) ConvertToVersion(in runtime.Object, _ string) (runtime.Object, error) {
	return in, nil
}

var validCodec = fakeCodec{}
var validAccessor = resourceAccessor{}
var validConvertor = fakeConvertor{}

func fakeInterfaces(version string) (*VersionInterfaces, bool) {
	return &VersionInterfaces{Codec: validCodec, ObjectConvertor: validConvertor, MetadataAccessor: validAccessor}, true
}

func unmatchedVersionInterfaces(version string) (*VersionInterfaces, bool) {
	return nil, false
}

func TestRESTMapperVersionAndKindForResource(t *testing.T) {
	testCases := []struct {
		Resource         string
		Kind, APIVersion string
		MixedCase        bool
		Err              bool
	}{
		{Resource: "internalobjec", Err: true},
		{Resource: "internalObjec", Err: true},

		{Resource: "internalobject", Kind: "InternalObject", APIVersion: "test"},
		{Resource: "internalobjects", Kind: "InternalObject", APIVersion: "test"},

		{Resource: "internalobject", MixedCase: true, Kind: "InternalObject", APIVersion: "test"},
		{Resource: "internalobjects", MixedCase: true, Kind: "InternalObject", APIVersion: "test"},

		{Resource: "internalObject", MixedCase: true, Kind: "InternalObject", APIVersion: "test"},
		{Resource: "internalObjects", MixedCase: true, Kind: "InternalObject", APIVersion: "test"},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]string{"test"}, fakeInterfaces)
		scheme := runtime.NewScheme()
		scheme.AddKnownTypes("test", &InternalObject{})
		mapper.Add(scheme, testCase.MixedCase, "test")

		v, k, err := mapper.VersionAndKindForResource(testCase.Resource)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %f: %v", i, testCase.Err, err)
			continue
		}
		if v != testCase.APIVersion || k != testCase.Kind {
			t.Errorf("%d: unexpected version and kind: %s %s", i, v, k)
		}
	}
}

func TestKindToResource(t *testing.T) {
	testCases := []struct {
		Kind             string
		MixedCase        bool
		Plural, Singular string
	}{
		{Kind: "Pod", MixedCase: true, Plural: "pods", Singular: "pod"},
		{Kind: "Pod", MixedCase: true, Plural: "pods", Singular: "pod"},
		{Kind: "Pod", MixedCase: false, Plural: "pods", Singular: "pod"},

		{Kind: "ReplicationController", MixedCase: true, Plural: "replicationControllers", Singular: "replicationController"},
		{Kind: "ReplicationController", MixedCase: true, Plural: "replicationControllers", Singular: "replicationController"},
		// API convention changed with regard to capitalization for v1beta3
		{Kind: "ReplicationController", MixedCase: false, Plural: "replicationcontrollers", Singular: "replicationcontroller"},

		{Kind: "lowercase", MixedCase: false, Plural: "lowercases", Singular: "lowercase"},
		// Don't add extra s if the original object is already plural
		{Kind: "lowercases", MixedCase: false, Plural: "lowercases", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		plural, singular := kindToResource(testCase.Kind, testCase.MixedCase)
		if singular != testCase.Singular || plural != testCase.Plural {
			t.Errorf("%d: unexpected plural and signular: %s %s", i, plural, singular)
		}
	}
}

func TestRESTMapperRESTMapping(t *testing.T) {
	testCases := []struct {
		Kind, APIVersion string
		MixedCase        bool

		Resource string
		Version  string
		Err      bool
	}{
		{Kind: "Unknown", APIVersion: "", Err: true},

		{Kind: "InternalObject", APIVersion: "test", Resource: "internalobjects"},
		{Kind: "InternalObject", APIVersion: "test", Resource: "internalobjects"},
		{Kind: "InternalObject", APIVersion: "", Resource: "internalobjects", Version: "test"},

		{Kind: "InternalObject", APIVersion: "test", Resource: "internalobjects"},
		{Kind: "InternalObject", APIVersion: "test", MixedCase: true, Resource: "internalObjects"},

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]string{"test"}, fakeInterfaces)
		scheme := runtime.NewScheme()
		scheme.AddKnownTypes("test", &InternalObject{})
		mapper.Add(scheme, testCase.MixedCase, "test")

		mapping, err := mapper.RESTMapping(testCase.APIVersion, testCase.Kind)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %f: %v", i, testCase.Err, err)
		}
		if hasErr {
			continue
		}
		if mapping.Resource != testCase.Resource {
			t.Errorf("%d: unexpected resource: %#v", i, mapping)
		}
		version := testCase.Version
		if version == "" {
			version = testCase.APIVersion
		}
		if mapping.APIVersion != version {
			t.Errorf("%d: unexpected version: %#v", i, mapping)
		}
		if mapping.Codec == nil || mapping.MetadataAccessor == nil || mapping.ObjectConvertor == nil {
			t.Errorf("%d: missing codec and accessor: %#v", i, mapping)
		}
	}
}

func TestRESTMapperRESTMappingSelectsVersion(t *testing.T) {
	mapper := NewDefaultRESTMapper([]string{"test1", "test2"}, fakeInterfaces)
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes("test1", &InternalObject{})
	scheme.AddKnownTypeWithName("test2", "OtherObject", &InternalObject{})
	scheme.AddKnownTypeWithName("test3", "OtherObject", &InternalObject{})
	mapper.Add(scheme, false, "test1", "test2")

	// pick default matching object kind based on search order
	mapping, err := mapper.RESTMapping("", "OtherObject")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if mapping.Resource != "otherobjects" || mapping.APIVersion != "test2" {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	mapping, err = mapper.RESTMapping("", "InternalObject")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if mapping.Resource != "internalobjects" || mapping.APIVersion != "test1" {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	// mismatch of version
	mapping, err = mapper.RESTMapping("test2", "InternalObject")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	mapping, err = mapper.RESTMapping("test1", "OtherObject")
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// not in the search versions
	mapping, err = mapper.RESTMapping("test3", "OtherObject")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestRESTMapperReportsErrorOnBadVersion(t *testing.T) {
	mapper := NewDefaultRESTMapper([]string{"test1", "test2"}, unmatchedVersionInterfaces)
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes("test1", &InternalObject{})
	mapper.Add(scheme, false, "test1")

	_, err := mapper.RESTMapping("test1", "InternalObject")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
