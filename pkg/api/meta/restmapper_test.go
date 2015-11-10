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
	"io"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type fakeCodec struct{}

func (fakeCodec) Encode(runtime.Object) ([]byte, error) {
	return []byte{}, nil
}

func (fakeCodec) EncodeToStream(runtime.Object, io.Writer) error {
	return nil
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

func (fakeCodec) DecodeParametersInto(parameters url.Values, obj runtime.Object) error {
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
	testGroup := "test.group"
	testVersion := "test"
	testGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: testVersion}

	testCases := []struct {
		Resource               string
		GroupVersionToRegister unversioned.GroupVersion
		ExpectedGVK            unversioned.GroupVersionKind
		MixedCase              bool
		Err                    bool
	}{
		{Resource: "internalobjec", Err: true, GroupVersionToRegister: testGroupVersion},
		{Resource: "internalObjec", Err: true, GroupVersionToRegister: testGroupVersion},

		{Resource: "internalobject", GroupVersionToRegister: testGroupVersion, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},
		{Resource: "internalobjects", GroupVersionToRegister: testGroupVersion, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},

		{Resource: "internalobject", GroupVersionToRegister: testGroupVersion, MixedCase: true, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},
		{Resource: "internalobjects", GroupVersionToRegister: testGroupVersion, MixedCase: true, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},

		{Resource: "internalObject", GroupVersionToRegister: testGroupVersion, MixedCase: true, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},
		{Resource: "internalObjects", GroupVersionToRegister: testGroupVersion, MixedCase: true, ExpectedGVK: unversioned.NewGroupVersionKind(testGroupVersion, "InternalObject")},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testGroup, []string{testGroupVersion.String()}, fakeInterfaces)
		mapper.Add(RESTScopeNamespace, testCase.ExpectedGVK.Kind, testCase.GroupVersionToRegister.String(), testCase.MixedCase)
		v, k, err := mapper.VersionAndKindForResource(testCase.Resource)

		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %t: %v", i, testCase.Err, err)
			continue
		}
		if err != nil {
			continue
		}

		actualGV, err := unversioned.ParseGroupVersion(v)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		actualGVK := unversioned.NewGroupVersionKind(actualGV, k)

		if actualGVK != testCase.ExpectedGVK {
			t.Errorf("%d: unexpected version and kind: e=%s a=%s", i, testCase.ExpectedGVK, actualGVK)
		}
	}
}

func TestRESTMapperGroupForResource(t *testing.T) {
	testCases := []struct {
		Resource     string
		Kind         string
		GroupVersion unversioned.GroupVersion
		Err          bool
	}{
		{Resource: "myObject", Kind: "MyObject", GroupVersion: unversioned.GroupVersion{Group: "testapi", Version: "test"}},
		{Resource: "myobject", Kind: "MyObject", GroupVersion: unversioned.GroupVersion{Group: "testapi2", Version: "test"}},
		{Resource: "myObje", Err: true, Kind: "MyObject", GroupVersion: unversioned.GroupVersion{Group: "testapi", Version: "test"}},
		{Resource: "myobje", Err: true, Kind: "MyObject", GroupVersion: unversioned.GroupVersion{Group: "testapi", Version: "test"}},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.GroupVersion.Group, []string{testCase.GroupVersion.String()}, fakeInterfaces)
		mapper.Add(RESTScopeNamespace, testCase.Kind, testCase.GroupVersion.String(), false)
		g, err := mapper.GroupForResource(testCase.Resource)
		if testCase.Err {
			if err == nil {
				t.Errorf("%d: expected error", i)
			}
		} else if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		} else if g != testCase.GroupVersion.Group {
			t.Errorf("%d: expected group %q, got %q", i, testCase.GroupVersion.Group, g)
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

		// Add "ies" when ending with "y"
		{Kind: "ImageRepository", MixedCase: true, Plural: "imageRepositories", Singular: "imageRepository"},
		// Add "es" when ending with "s"
		{Kind: "miss", MixedCase: false, Plural: "misses", Singular: "miss"},
		// Add "s" otherwise
		{Kind: "lowercase", MixedCase: false, Plural: "lowercases", Singular: "lowercase"},
	}
	for i, testCase := range testCases {
		plural, singular := KindToResource(testCase.Kind, testCase.MixedCase)
		if singular != testCase.Singular || plural != testCase.Plural {
			t.Errorf("%d: unexpected plural and singular: %s %s", i, plural, singular)
		}
	}
}

func TestRESTMapperResourceSingularizer(t *testing.T) {
	testCases := []struct {
		Kind      string
		MixedCase bool
		Plural    string
		Singular  string
	}{
		{Kind: "Pod", MixedCase: true, Plural: "pods", Singular: "pod"},
		{Kind: "Pod", MixedCase: false, Plural: "pods", Singular: "pod"},

		{Kind: "ReplicationController", MixedCase: true, Plural: "replicationControllers", Singular: "replicationController"},
		{Kind: "ReplicationController", MixedCase: false, Plural: "replicationcontrollers", Singular: "replicationcontroller"},

		{Kind: "ImageRepository", MixedCase: true, Plural: "imageRepositories", Singular: "imageRepository"},
		{Kind: "ImageRepository", MixedCase: false, Plural: "imagerepositories", Singular: "imagerepository"},

		{Kind: "Status", MixedCase: true, Plural: "statuses", Singular: "status"},
		{Kind: "Status", MixedCase: false, Plural: "statuses", Singular: "status"},

		{Kind: "lowercase", MixedCase: false, Plural: "lowercases", Singular: "lowercase"},
		// Don't add extra s if the original object is already plural
		{Kind: "lowercases", MixedCase: false, Plural: "lowercases", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		testGroupVersion := unversioned.GroupVersion{Group: "tgroup", Version: "test"}

		mapper := NewDefaultRESTMapper(testGroupVersion.Group, []string{testGroupVersion.String()}, fakeInterfaces)
		// create singular/plural mapping
		mapper.Add(RESTScopeNamespace, testCase.Kind, testGroupVersion.String(), testCase.MixedCase)

		singular, _ := mapper.ResourceSingularizer(testCase.Plural)
		if singular != testCase.Singular {
			t.Errorf("%d: mismatched singular: %s, should be %s", i, singular, testCase.Singular)
		}
	}
}

func TestRESTMapperRESTMapping(t *testing.T) {
	testGroup := "tgroup"
	testGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: "test"}

	testCases := []struct {
		Kind             string
		APIGroupVersions []unversioned.GroupVersion
		MixedCase        bool
		DefaultVersions  []string

		Resource             string
		ExpectedGroupVersion *unversioned.GroupVersion
		Err                  bool
	}{
		{Kind: "Unknown", Err: true},
		{Kind: "InternalObject", Err: true},

		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "Unknown", Err: true},

		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},
		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},

		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},

		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{}, Resource: "internalobjects", ExpectedGroupVersion: &unversioned.GroupVersion{Group: testGroup, Version: "test"}},

		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},
		{DefaultVersions: []string{testGroupVersion.String()}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, MixedCase: true, Resource: "internalObjects"},

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		internalGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: "test"}

		mapper := NewDefaultRESTMapper(testGroup, testCase.DefaultVersions, fakeInterfaces)
		mapper.Add(RESTScopeNamespace, "InternalObject", internalGroupVersion.String(), testCase.MixedCase)

		deprecatedGroupVersionStrings := []string{}
		for _, gv := range testCase.APIGroupVersions {
			deprecatedGroupVersionStrings = append(deprecatedGroupVersionStrings, gv.String())
		}

		mapping, err := mapper.RESTMapping(testCase.Kind, deprecatedGroupVersionStrings...)
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

		if mapping.Codec == nil || mapping.MetadataAccessor == nil || mapping.ObjectConvertor == nil {
			t.Errorf("%d: missing codec and accessor: %#v", i, mapping)
		}

		groupVersion := testCase.ExpectedGroupVersion
		if groupVersion == nil {
			groupVersion = &testCase.APIGroupVersions[0]
		}
		if mapping.GroupVersionKind.GroupVersion() != *groupVersion {
			t.Errorf("%d: unexpected version: %#v", i, mapping)
		}

	}
}

func TestRESTMapperRESTMappingSelectsVersion(t *testing.T) {
	expectedGroupVersion1 := unversioned.GroupVersion{Group: "tgroup", Version: "test1"}
	expectedGroupVersion2 := unversioned.GroupVersion{Group: "tgroup", Version: "test2"}
	expectedGroupVersion3 := unversioned.GroupVersion{Group: "tgroup", Version: "test3"}

	mapper := NewDefaultRESTMapper("tgroup", []string{expectedGroupVersion1.String(), expectedGroupVersion2.String()}, fakeInterfaces)
	mapper.Add(RESTScopeNamespace, "InternalObject", expectedGroupVersion1.String(), false)
	mapper.Add(RESTScopeNamespace, "OtherObject", expectedGroupVersion2.String(), false)

	// pick default matching object kind based on search order
	mapping, err := mapper.RESTMapping("OtherObject")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "otherobjects" || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion2 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	mapping, err = mapper.RESTMapping("InternalObject")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "internalobjects" || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion1 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	// mismatch of version
	mapping, err = mapper.RESTMapping("InternalObject", expectedGroupVersion2.String())
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	mapping, err = mapper.RESTMapping("OtherObject", expectedGroupVersion1.String())
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// not in the search versions
	mapping, err = mapper.RESTMapping("OtherObject", expectedGroupVersion3.String())
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// explicit search order
	mapping, err = mapper.RESTMapping("OtherObject", expectedGroupVersion3.String(), expectedGroupVersion1.String())
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	mapping, err = mapper.RESTMapping("OtherObject", expectedGroupVersion3.String(), expectedGroupVersion2.String())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "otherobjects" || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion2 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}
}

func TestRESTMapperReportsErrorOnBadVersion(t *testing.T) {
	expectedGroupVersion1 := unversioned.GroupVersion{Group: "tgroup", Version: "test1"}
	expectedGroupVersion2 := unversioned.GroupVersion{Group: "tgroup", Version: "test2"}

	mapper := NewDefaultRESTMapper("tgroup", []string{expectedGroupVersion1.String(), expectedGroupVersion2.String()}, unmatchedVersionInterfaces)
	mapper.Add(RESTScopeNamespace, "InternalObject", expectedGroupVersion1.String(), false)
	_, err := mapper.RESTMapping("InternalObject", expectedGroupVersion1.String())
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
