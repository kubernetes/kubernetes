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

package meta

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type fakeConvertor struct{}

func (fakeConvertor) Convert(in, out, context interface{}) error {
	return nil
}

func (fakeConvertor) ConvertToVersion(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
	return in, nil
}

func (fakeConvertor) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return label, value, nil
}

var validAccessor = resourceAccessor{}
var validConvertor = fakeConvertor{}

func fakeInterfaces(version unversioned.GroupVersion) (*VersionInterfaces, error) {
	return &VersionInterfaces{ObjectConvertor: validConvertor, MetadataAccessor: validAccessor}, nil
}

var unmatchedErr = errors.New("no version")

func unmatchedVersionInterfaces(version unversioned.GroupVersion) (*VersionInterfaces, error) {
	return nil, unmatchedErr
}

func TestRESTMapperVersionAndKindForResource(t *testing.T) {
	testGroup := "test.group"
	testVersion := "test"
	testGroupVersion := unversioned.GroupVersion{Group: testGroup, Version: testVersion}

	testCases := []struct {
		Resource               unversioned.GroupVersionResource
		GroupVersionToRegister unversioned.GroupVersion
		ExpectedGVK            unversioned.GroupVersionKind
		Err                    bool
	}{
		{Resource: unversioned.GroupVersionResource{Resource: "internalobjec"}, Err: true},
		{Resource: unversioned.GroupVersionResource{Resource: "internalObjec"}, Err: true},

		{Resource: unversioned.GroupVersionResource{Resource: "internalobject"}, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
		{Resource: unversioned.GroupVersionResource{Resource: "internalobjects"}, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testGroupVersion}, fakeInterfaces)
		if len(testCase.ExpectedGVK.Kind) != 0 {
			mapper.Add(testCase.ExpectedGVK, RESTScopeNamespace)
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
		Resource         unversioned.GroupVersionResource
		GroupVersionKind unversioned.GroupVersionKind
		Err              bool
	}{
		{Resource: unversioned.GroupVersionResource{Resource: "myObject"}, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: unversioned.GroupVersionResource{Resource: "myobject"}, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi2", Version: "test", Kind: "MyObject"}},
		{Resource: unversioned.GroupVersionResource{Resource: "myObje"}, Err: true, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: unversioned.GroupVersionResource{Resource: "myobje"}, Err: true, GroupVersionKind: unversioned.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testCase.GroupVersionKind.GroupVersion()}, fakeInterfaces)
		mapper.Add(testCase.GroupVersionKind, RESTScopeNamespace)

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

func TestRESTMapperKindsFor(t *testing.T) {
	testCases := []struct {
		Name                     string
		PreferredOrder           []unversioned.GroupVersion
		KindsToRegister          []unversioned.GroupVersionKind
		PartialResourceToRequest unversioned.GroupVersionResource

		ExpectedKinds   []unversioned.GroupVersionKind
		ExpectedKindErr string
	}{
		{
			Name: "ambiguous groups, with preference order",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: unversioned.GroupVersionResource{Resource: "my-kinds"},

			ExpectedKinds: []unversioned.GroupVersionKind{
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			ExpectedKindErr: " matches multiple kinds ",
		},

		{
			Name: "ambiguous groups, with explicit group match",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: unversioned.GroupVersionResource{Group: "first-group", Resource: "my-kinds"},

			ExpectedKinds: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
		},

		{
			Name: "ambiguous groups, with ambiguous version match",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "first-group", Version: "first-version"},
				{Group: "second-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: unversioned.GroupVersionResource{Version: "first-version", Resource: "my-kinds"},

			ExpectedKinds: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
			},
			ExpectedKindErr: " matches multiple kinds ",
		},
	}
	for _, testCase := range testCases {
		tcName := testCase.Name
		mapper := NewDefaultRESTMapper(testCase.PreferredOrder, fakeInterfaces)
		for _, kind := range testCase.KindsToRegister {
			mapper.Add(kind, RESTScopeNamespace)
		}

		actualKinds, err := mapper.KindsFor(testCase.PartialResourceToRequest)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tcName, err)
			continue
		}
		if !reflect.DeepEqual(testCase.ExpectedKinds, actualKinds) {
			t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedKinds, actualKinds)
		}

		singleKind, err := mapper.KindFor(testCase.PartialResourceToRequest)
		if err == nil && len(testCase.ExpectedKindErr) != 0 {
			t.Errorf("%s: expected error: %v", tcName, testCase.ExpectedKindErr)
			continue
		}
		if err != nil {
			if len(testCase.ExpectedKindErr) == 0 {
				t.Errorf("%s: unexpected error: %v", tcName, err)
				continue
			} else {
				if !strings.Contains(err.Error(), testCase.ExpectedKindErr) {
					t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedKindErr, err)
					continue
				}
			}

		} else {
			if testCase.ExpectedKinds[0] != singleKind {
				t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedKinds[0], singleKind)
			}

		}
	}
}

func TestRESTMapperResourcesFor(t *testing.T) {
	testCases := []struct {
		Name                             string
		PreferredOrder                   []unversioned.GroupVersion
		KindsToRegister                  []unversioned.GroupVersionKind
		PluralPartialResourceToRequest   unversioned.GroupVersionResource
		SingularPartialResourceToRequest unversioned.GroupVersionResource

		ExpectedResources   []unversioned.GroupVersionResource
		ExpectedResourceErr string
	}{
		{
			Name: "ambiguous groups, with preference order",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   unversioned.GroupVersionResource{Resource: "my-kinds"},
			SingularPartialResourceToRequest: unversioned.GroupVersionResource{Resource: "my-kind"},

			ExpectedResources: []unversioned.GroupVersionResource{
				{Group: "second-group", Version: "first-version", Resource: "my-kinds"},
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
			ExpectedResourceErr: " matches multiple resources ",
		},

		{
			Name: "ambiguous groups, with explicit group match",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   unversioned.GroupVersionResource{Group: "first-group", Resource: "my-kinds"},
			SingularPartialResourceToRequest: unversioned.GroupVersionResource{Group: "first-group", Resource: "my-kind"},

			ExpectedResources: []unversioned.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
		},

		{
			Name: "ambiguous groups, with ambiguous version match",
			PreferredOrder: []unversioned.GroupVersion{
				{Group: "first-group", Version: "first-version"},
				{Group: "second-group", Version: "first-version"},
			},
			KindsToRegister: []unversioned.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   unversioned.GroupVersionResource{Version: "first-version", Resource: "my-kinds"},
			SingularPartialResourceToRequest: unversioned.GroupVersionResource{Version: "first-version", Resource: "my-kind"},

			ExpectedResources: []unversioned.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
				{Group: "second-group", Version: "first-version", Resource: "my-kinds"},
			},
			ExpectedResourceErr: " matches multiple resources ",
		},
	}
	for _, testCase := range testCases {
		tcName := testCase.Name

		for _, partialResource := range []unversioned.GroupVersionResource{testCase.PluralPartialResourceToRequest, testCase.SingularPartialResourceToRequest} {
			mapper := NewDefaultRESTMapper(testCase.PreferredOrder, fakeInterfaces)
			for _, kind := range testCase.KindsToRegister {
				mapper.Add(kind, RESTScopeNamespace)
			}

			actualResources, err := mapper.ResourcesFor(partialResource)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tcName, err)
				continue
			}
			if !reflect.DeepEqual(testCase.ExpectedResources, actualResources) {
				t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedResources, actualResources)
			}

			singleResource, err := mapper.ResourceFor(partialResource)
			if err == nil && len(testCase.ExpectedResourceErr) != 0 {
				t.Errorf("%s: expected error: %v", tcName, testCase.ExpectedResourceErr)
				continue
			}
			if err != nil {
				if len(testCase.ExpectedResourceErr) == 0 {
					t.Errorf("%s: unexpected error: %v", tcName, err)
					continue
				} else {
					if !strings.Contains(err.Error(), testCase.ExpectedResourceErr) {
						t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedResourceErr, err)
						continue
					}
				}

			} else {
				if testCase.ExpectedResources[0] != singleResource {
					t.Errorf("%s: expected %v, got %v", tcName, testCase.ExpectedResources[0], singleResource)
				}

			}
		}
	}
}

func TestKindToResource(t *testing.T) {
	testCases := []struct {
		Kind             string
		Plural, Singular string
	}{
		{Kind: "Pod", Plural: "pods", Singular: "pod"},

		{Kind: "ReplicationController", Plural: "replicationcontrollers", Singular: "replicationcontroller"},

		// Add "ies" when ending with "y"
		{Kind: "ImageRepository", Plural: "imagerepositories", Singular: "imagerepository"},
		// Add "es" when ending with "s"
		{Kind: "miss", Plural: "misses", Singular: "miss"},
		// Add "s" otherwise
		{Kind: "lowercase", Plural: "lowercases", Singular: "lowercase"},
	}
	for i, testCase := range testCases {
		version := unversioned.GroupVersion{}

		plural, singular := KindToResource(version.WithKind(testCase.Kind))
		if singular != version.WithResource(testCase.Singular) || plural != version.WithResource(testCase.Plural) {
			t.Errorf("%d: unexpected plural and singular: %v %v", i, plural, singular)
		}
	}
}

func TestRESTMapperResourceSingularizer(t *testing.T) {
	testGroupVersion := unversioned.GroupVersion{Group: "tgroup", Version: "test"}

	testCases := []struct {
		Kind     string
		Plural   string
		Singular string
	}{
		{Kind: "Pod", Plural: "pods", Singular: "pod"},
		{Kind: "ReplicationController", Plural: "replicationcontrollers", Singular: "replicationcontroller"},
		{Kind: "ImageRepository", Plural: "imagerepositories", Singular: "imagerepository"},
		{Kind: "Status", Plural: "statuses", Singular: "status"},

		{Kind: "lowercase", Plural: "lowercases", Singular: "lowercase"},
		// TODO this test is broken.  This updates to reflect actual behavior.  Kinds are expected to be singular
		// old (incorrect), coment: Don't add extra s if the original object is already plural
		{Kind: "lowercases", Plural: "lowercaseses", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]unversioned.GroupVersion{testGroupVersion}, fakeInterfaces)
		// create singular/plural mapping
		mapper.Add(testGroupVersion.WithKind(testCase.Kind), RESTScopeNamespace)

		singular, err := mapper.ResourceSingularizer(testCase.Plural)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if singular != testCase.Singular {
			t.Errorf("%d: mismatched singular: got %v, expected %v", i, singular, testCase.Singular)
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

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.DefaultVersions, fakeInterfaces)
		mapper.Add(internalGroupVersion.WithKind("InternalObject"), RESTScopeNamespace)

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

		if mapping.MetadataAccessor == nil || mapping.ObjectConvertor == nil {
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
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace)
	mapper.Add(expectedGroupVersion2.WithKind("OtherObject"), RESTScopeNamespace)

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
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace)
	_, err := mapper.RESTMapping(internalObjectGK, expectedGroupVersion1.Version)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
