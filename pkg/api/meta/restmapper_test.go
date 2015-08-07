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

package meta

import (
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/runtime"
)

type fakeCodec struct{}

func (fakeCodec) Encode(runtime.Object) ([]byte, error) {
	return []byte{}, nil
}

func (fakeCodec) Decode([]byte) (runtime.Object, error) {
	return nil, nil
}

func (fakeCodec) DecodeToVersion([]byte, string) (runtime.Object, error) {
	return nil, nil
}

func (fakeCodec) DecodeInto([]byte, runtime.Object) error {
	return nil
}

func (fakeCodec) DecodeIntoWithSpecifiedVersionKind([]byte, runtime.Object, string, string) error {
	return nil
}

type fakeConvertor struct{}

func (fakeConvertor) Convert(in, out interface{}) error {
	return nil
}

func (fakeConvertor) ConvertToVersion(in runtime.Object, _ string) (runtime.Object, error) {
	return in, nil
}

func (fakeConvertor) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return label, value, nil
}

var validCodec = fakeCodec{}
var validAccessor = resourceAccessor{}
var validConvertor = fakeConvertor{}

func fakeInterfaces(version string) (*VersionInterfaces, error) {
	return &VersionInterfaces{Codec: validCodec, ObjectConvertor: validConvertor, MetadataAccessor: validAccessor}, nil
}

var unmatchedErr = errors.New("no version")

func unmatchedVersionInterfaces(version string) (*VersionInterfaces, error) {
	return nil, unmatchedErr
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
		mapper.Add(RESTScopeNamespace, testCase.Kind, testCase.APIVersion, testCase.MixedCase)
		v, k, err := mapper.VersionAndKindForResource(testCase.Resource)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %t: %v", i, testCase.Err, err)
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
		{Kind: "ReplicationController", MixedCase: false, Plural: "replicationcontrollers", Singular: "replicationcontroller"},

		{Kind: "ImageRepository", MixedCase: true, Plural: "imageRepositories", Singular: "imageRepository"},

		{Kind: "lowercase", MixedCase: false, Plural: "lowercases", Singular: "lowercase"},
		// Don't add extra s if the original object is already plural
		{Kind: "lowercases", MixedCase: false, Plural: "lowercases", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		plural, singular := kindToResource(testCase.Kind, testCase.MixedCase)
		if singular != testCase.Singular || plural != testCase.Plural {
			t.Errorf("%d: unexpected plural and singular: %s %s", i, plural, singular)
		}
	}
}

func TestRESTMapperResourceSingularizer(t *testing.T) {
	testCases := []struct {
		Kind, APIVersion string
		MixedCase        bool
		Plural           string
		Singular         string
	}{
		{Kind: "Pod", APIVersion: "test", MixedCase: true, Plural: "pods", Singular: "pod"},
		{Kind: "Pod", APIVersion: "test", MixedCase: false, Plural: "pods", Singular: "pod"},

		{Kind: "ReplicationController", APIVersion: "test", MixedCase: true, Plural: "replicationControllers", Singular: "replicationController"},
		{Kind: "ReplicationController", APIVersion: "test", MixedCase: false, Plural: "replicationcontrollers", Singular: "replicationcontroller"},

		{Kind: "ImageRepository", APIVersion: "test", MixedCase: true, Plural: "imageRepositories", Singular: "imageRepository"},
		{Kind: "ImageRepository", APIVersion: "test", MixedCase: false, Plural: "imagerepositories", Singular: "imagerepository"},

		{Kind: "Status", APIVersion: "test", MixedCase: true, Plural: "statuses", Singular: "status"},
		{Kind: "Status", APIVersion: "test", MixedCase: false, Plural: "statuses", Singular: "status"},

		{Kind: "lowercase", APIVersion: "test", MixedCase: false, Plural: "lowercases", Singular: "lowercase"},
		// Don't add extra s if the original object is already plural
		{Kind: "lowercases", APIVersion: "test", MixedCase: false, Plural: "lowercases", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]string{"test"}, fakeInterfaces)
		// create singular/plural mapping
		mapper.Add(RESTScopeNamespace, testCase.Kind, testCase.APIVersion, testCase.MixedCase)
		singular, _ := mapper.ResourceSingularizer(testCase.Plural)
		if singular != testCase.Singular {
			t.Errorf("%d: mismatched singular: %s, should be %s", i, singular, testCase.Singular)
		}
	}
}

func TestRESTMapperRESTMapping(t *testing.T) {
	testCases := []struct {
		Kind            string
		APIVersions     []string
		MixedCase       bool
		DefaultVersions []string

		Resource string
		Version  string
		Err      bool
	}{
		{Kind: "Unknown", Err: true},
		{Kind: "InternalObject", Err: true},

		{DefaultVersions: []string{"test"}, Kind: "Unknown", Err: true},

		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{"test"}, Resource: "internalobjects"},
		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{"test"}, Resource: "internalobjects"},

		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{"test"}, Resource: "internalobjects"},

		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{}, Resource: "internalobjects", Version: "test"},

		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{"test"}, Resource: "internalobjects"},
		{DefaultVersions: []string{"test"}, Kind: "InternalObject", APIVersions: []string{"test"}, MixedCase: true, Resource: "internalObjects"},

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.DefaultVersions, fakeInterfaces)
		mapper.Add(RESTScopeNamespace, "InternalObject", "test", testCase.MixedCase)
		mapping, err := mapper.RESTMapping(testCase.Kind, testCase.APIVersions...)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %t: %v", i, testCase.Err, err)
		}
		if hasErr {
			continue
		}
		if mapping.Resource != testCase.Resource {
			t.Errorf("%d: unexpected resource: %#v", i, mapping)
		}
		version := testCase.Version
		if version == "" {
			version = testCase.APIVersions[0]
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
	mapper.Add(RESTScopeNamespace, "InternalObject", "test1", false)
	mapper.Add(RESTScopeNamespace, "OtherObject", "test2", false)

	// pick default matching object kind based on search order
	mapping, err := mapper.RESTMapping("OtherObject")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "otherobjects" || mapping.APIVersion != "test2" {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	mapping, err = mapper.RESTMapping("InternalObject")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "internalobjects" || mapping.APIVersion != "test1" {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	// mismatch of version
	mapping, err = mapper.RESTMapping("InternalObject", "test2")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	mapping, err = mapper.RESTMapping("OtherObject", "test1")
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// not in the search versions
	mapping, err = mapper.RESTMapping("OtherObject", "test3")
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// explicit search order
	mapping, err = mapper.RESTMapping("OtherObject", "test3", "test1")
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	mapping, err = mapper.RESTMapping("OtherObject", "test3", "test2")
	if err != nil {
		t.Fatalf("unexpected non-error")
	}
	if mapping.Resource != "otherobjects" || mapping.APIVersion != "test2" {
		t.Errorf("unexpected mapping: %#v", mapping)
	}
}

func TestRESTMapperReportsErrorOnBadVersion(t *testing.T) {
	mapper := NewDefaultRESTMapper([]string{"test1", "test2"}, unmatchedVersionInterfaces)
	mapper.Add(RESTScopeNamespace, "InternalObject", "test1", false)
	_, err := mapper.RESTMapping("InternalObject", "test1")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
