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

var _ runtime.Decoder = fakeCodec{}

func (fakeCodec) Encode(runtime.Object) ([]byte, error) {
	return []byte{}, nil
}

func (fakeCodec) EncodeToStream(runtime.Object, io.Writer) error {
	return nil
}

func (fakeCodec) Decode([]byte) (runtime.Object, error) {
	return nil, nil
}

func (fakeCodec) DecodeToVersion([]byte, unversioned.GroupVersion) (runtime.Object, error) {
	return nil, nil
}

func (fakeCodec) DecodeInto([]byte, runtime.Object) error {
	return nil
}

func (fakeCodec) DecodeIntoWithSpecifiedVersionKind([]byte, runtime.Object, unversioned.GroupVersionKind) error {
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
		{Resource: "internalobjec", Err: true},
		{Resource: "internalObjec", Err: true},

		{Resource: "internalobject", ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
		{Resource: "internalobjects", ExpectedGVK: testGroupVersion.WithKind("InternalObject")},

		{Resource: "internalobject", MixedCase: true, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
		{Resource: "internalobjects", MixedCase: true, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},

		{Resource: "internalObject", MixedCase: true, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
		{Resource: "internalObjects", MixedCase: true, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testGroupVersion}, fakeInterfaces)
		if len(testCase.ExpectedGVK.Kind) != 0 {
			mapper.Add(testCase.ExpectedGVK, RESTScopeNamespace, testCase.MixedCase)
		}
		actualGVK, err := mapper.KindFor(testCase.Resource)

		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %t: %v", i, testCase.Err, err)
			continue
		}
		if err != nil {
			continue
		}

		if actualGVK != testCase.ExpectedGVK {
			t.Errorf("%d: unexpected version and kind: e=%s a=%s", i, testCase.ExpectedGVK, actualGVK)
		}
	}
}

func TestRESTMapperGroupForResource(t *testing.T) {
	testCases := []struct {
		Resource         string
		GroupVersionKind unversioned.GroupVersionKind
		Err              bool
	}{
		{Resource: "myObject", GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: "myobject", GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi2", Version: "test", Kind: "MyObject"}},
		{Resource: "myObje", Err: true, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: "myobje", Err: true, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testCase.GroupVersionKind.GroupVersion()}, fakeInterfaces)
		mapper.Add(testCase.GroupVersionKind, RESTScopeNamespace, false)
		actualGVK, err := mapper.KindFor(testCase.Resource)
		if testCase.Err {
			if err == nil {
				t.Errorf("%d: expected error", i)
			}
		} else if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		} else if actualGVK != testCase.GroupVersionKind {
			t.Errorf("%d: expected group %q, got %q", i, testCase.GroupVersionKind, actualGVK)
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
	testGroupVersion := unversioned.GroupVersion{Group: "tgroup", Version: "test"}

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
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testGroupVersion}, fakeInterfaces)
		// create singular/plural mapping
		mapper.Add(testGroupVersion.WithKind(testCase.Kind), RESTScopeNamespace, testCase.MixedCase)

		singular, _ := mapper.ResourceSingularizer(testCase.Plural)
		if singular != testCase.Singular {
			t.Errorf("%d: mismatched singular: %s, should be %s", i, singular, testCase.Singular)
		}
	}
}

func TestRESTMapperRESTMapping(t *testing.T) {
	testGroup := "tgroup"
	testGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: "test"}
	internalGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: "test"}

	testCases := []struct {
		Kind             string
		APIGroupVersions []unversioned.GroupVersion
		MixedCase        bool
		DefaultVersions  []unversioned.GroupVersion

		Resource             string
		ExpectedGroupVersion *unversioned.GroupVersion
		Err                  bool
	}{
		{Kind: "Unknown", Err: true},
		{Kind: "InternalObject", Err: true},

		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "Unknown", Err: true},

		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},
		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},

		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},

		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{}, Resource: "internalobjects", ExpectedGroupVersion: &unversioned.GroupVersion{Group: testGroup, Version: "test"}},

		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: "internalobjects"},
		{DefaultVersions: []unversioned.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []unversioned.GroupVersion{{Group: testGroup, Version: "test"}}, MixedCase: true, Resource: "internalObjects"},

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.DefaultVersions, fakeInterfaces)
		mapper.Add(internalGroupVersion.WithKind("InternalObject"), RESTScopeNamespace, testCase.MixedCase)

		preferredVersions := []string{}
		for _, gv := range testCase.APIGroupVersions {
			preferredVersions = append(preferredVersions, gv.Version)
		}
		gk := unversioned.GroupKind{Group: testGroup, Kind: testCase.Kind}

		mapping, err := mapper.RESTMapping(gk, preferredVersions...)
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
	internalObjectGK := unversioned.GroupKind{Group: "tgroup", Kind: "InternalObject"}
	otherObjectGK := unversioned.GroupKind{Group: "tgroup", Kind: "OtherObject"}

	mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{expectedGroupVersion1, expectedGroupVersion2}, fakeInterfaces)
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace, false)
	mapper.Add(expectedGroupVersion2.WithKind("OtherObject"), RESTScopeNamespace, false)

	// pick default matching object kind based on search order
	mapping, err := mapper.RESTMapping(otherObjectGK)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "otherobjects" || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion2 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	mapping, err = mapper.RESTMapping(internalObjectGK)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "internalobjects" || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion1 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	// mismatch of version
	mapping, err = mapper.RESTMapping(internalObjectGK, expectedGroupVersion2.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	mapping, err = mapper.RESTMapping(otherObjectGK, expectedGroupVersion1.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// not in the search versions
	mapping, err = mapper.RESTMapping(otherObjectGK, expectedGroupVersion3.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	// explicit search order
	mapping, err = mapper.RESTMapping(otherObjectGK, expectedGroupVersion3.Version, expectedGroupVersion1.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}

	mapping, err = mapper.RESTMapping(otherObjectGK, expectedGroupVersion3.Version, expectedGroupVersion2.Version)
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
	internalObjectGK := unversioned.GroupKind{Group: "tgroup", Kind: "InternalObject"}

	mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{expectedGroupVersion1, expectedGroupVersion2}, unmatchedVersionInterfaces)
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace, false)
	_, err := mapper.RESTMapping(internalObjectGK, expectedGroupVersion1.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
