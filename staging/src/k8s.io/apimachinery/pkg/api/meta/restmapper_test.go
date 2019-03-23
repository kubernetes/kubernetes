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
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestRESTMapperVersionAndKindForResource(t *testing.T) {
	testGroup := "test.group"
	testVersion := "test"
	testGroupVersion := schema.GroupVersion{Group: testGroup, Version: testVersion}

	testCases := []struct {
		Resource               schema.GroupVersionResource
		GroupVersionToRegister schema.GroupVersion
		ExpectedGVK            schema.GroupVersionKind
		Err                    bool
	}{
		{Resource: schema.GroupVersionResource{Resource: "internalobjec"}, Err: true},
		{Resource: schema.GroupVersionResource{Resource: "internalObjec"}, Err: true},

		{Resource: schema.GroupVersionResource{Resource: "internalobject"}, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
		{Resource: schema.GroupVersionResource{Resource: "internalobjects"}, ExpectedGVK: testGroupVersion.WithKind("InternalObject")},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]schema.GroupVersion{testGroupVersion})
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
		Resource         schema.GroupVersionResource
		GroupVersionKind schema.GroupVersionKind
		Err              bool
	}{
		{Resource: schema.GroupVersionResource{Resource: "myObject"}, GroupVersionKind: schema.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: schema.GroupVersionResource{Resource: "myobject"}, GroupVersionKind: schema.GroupVersionKind{Group: "testapi2", Version: "test", Kind: "MyObject"}},
		{Resource: schema.GroupVersionResource{Resource: "myObje"}, Err: true, GroupVersionKind: schema.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
		{Resource: schema.GroupVersionResource{Resource: "myobje"}, Err: true, GroupVersionKind: schema.GroupVersionKind{Group: "testapi", Version: "test", Kind: "MyObject"}},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]schema.GroupVersion{testCase.GroupVersionKind.GroupVersion()})
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
		PreferredOrder           []schema.GroupVersion
		KindsToRegister          []schema.GroupVersionKind
		PartialResourceToRequest schema.GroupVersionResource

		ExpectedKinds   []schema.GroupVersionKind
		ExpectedKindErr string
	}{
		{
			// exact matches are preferred
			Name: "groups, with group exact",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group-1", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group-1", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Group: "first-group", Resource: "my-kind"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
		},

		{
			// group prefixes work
			Name: "groups, with group prefix",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Group: "first", Resource: "my-kind"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
		},

		{
			// group prefixes can be ambiguous
			Name: "groups, with ambiguous group prefix",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group-1", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group-1", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Group: "first", Resource: "my-kind"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "first-group-1", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			ExpectedKindErr: " matches multiple kinds ",
		},

		{
			Name: "ambiguous groups, with preference order",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Resource: "my-kinds"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			ExpectedKindErr: " matches multiple kinds ",
		},

		{
			Name: "ambiguous groups, with explicit group match",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Group: "first-group", Resource: "my-kinds"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
		},

		{
			Name: "ambiguous groups, with ambiguous version match",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group", Version: "first-version"},
				{Group: "second-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PartialResourceToRequest: schema.GroupVersionResource{Version: "first-version", Resource: "my-kinds"},

			ExpectedKinds: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
			},
			ExpectedKindErr: " matches multiple kinds ",
		},
	}
	for _, testCase := range testCases {
		tcName := testCase.Name
		mapper := NewDefaultRESTMapper(testCase.PreferredOrder)
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
		PreferredOrder                   []schema.GroupVersion
		KindsToRegister                  []schema.GroupVersionKind
		PluralPartialResourceToRequest   schema.GroupVersionResource
		SingularPartialResourceToRequest schema.GroupVersionResource

		ExpectedResources   []schema.GroupVersionResource
		ExpectedResourceErr string
	}{
		{
			// exact matches are preferred
			Name: "groups, with group exact",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group-1", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group-1", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Group: "first-group", Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Group: "first-group", Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
		},

		{
			// group prefixes work
			Name: "groups, with group prefix",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Group: "first", Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Group: "first", Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
		},

		{
			// group prefixes can be ambiguous
			Name: "groups, with ambiguous group prefix",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group-1", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group-1", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Group: "first", Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Group: "first", Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "first-group-1", Version: "first-version", Resource: "my-kinds"},
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
			ExpectedResourceErr: " matches multiple resources ",
		},

		{
			Name: "ambiguous groups, with preference order",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "second-group", Version: "first-version", Resource: "my-kinds"},
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
			ExpectedResourceErr: " matches multiple resources ",
		},

		{
			Name: "ambiguous groups, with explicit group match",
			PreferredOrder: []schema.GroupVersion{
				{Group: "second-group", Version: "first-version"},
				{Group: "first-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Group: "first-group", Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Group: "first-group", Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
			},
		},

		{
			Name: "ambiguous groups, with ambiguous version match",
			PreferredOrder: []schema.GroupVersion{
				{Group: "first-group", Version: "first-version"},
				{Group: "second-group", Version: "first-version"},
			},
			KindsToRegister: []schema.GroupVersionKind{
				{Group: "first-group", Version: "first-version", Kind: "my-kind"},
				{Group: "first-group", Version: "first-version", Kind: "your-kind"},
				{Group: "second-group", Version: "first-version", Kind: "my-kind"},
				{Group: "second-group", Version: "first-version", Kind: "your-kind"},
			},
			PluralPartialResourceToRequest:   schema.GroupVersionResource{Version: "first-version", Resource: "my-kinds"},
			SingularPartialResourceToRequest: schema.GroupVersionResource{Version: "first-version", Resource: "my-kind"},

			ExpectedResources: []schema.GroupVersionResource{
				{Group: "first-group", Version: "first-version", Resource: "my-kinds"},
				{Group: "second-group", Version: "first-version", Resource: "my-kinds"},
			},
			ExpectedResourceErr: " matches multiple resources ",
		},
	}
	for _, testCase := range testCases {
		tcName := testCase.Name

		for _, partialResource := range []schema.GroupVersionResource{testCase.PluralPartialResourceToRequest, testCase.SingularPartialResourceToRequest} {
			mapper := NewDefaultRESTMapper(testCase.PreferredOrder)
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
		version := schema.GroupVersion{}

		plural, singular := UnsafeGuessKindToResource(version.WithKind(testCase.Kind))
		if singular != version.WithResource(testCase.Singular) || plural != version.WithResource(testCase.Plural) {
			t.Errorf("%d: unexpected plural and singular: %v %v", i, plural, singular)
		}
	}
}

func TestRESTMapperResourceSingularizer(t *testing.T) {
	testGroupVersion := schema.GroupVersion{Group: "tgroup", Version: "test"}

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
		// old (incorrect), comment: Don't add extra s if the original object is already plural
		{Kind: "lowercases", Plural: "lowercaseses", Singular: "lowercases"},
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper([]schema.GroupVersion{testGroupVersion})
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
	testGroupVersion := schema.GroupVersion{Group: testGroup, Version: "test"}
	internalGroupVersion := schema.GroupVersion{Group: testGroup, Version: "test"}

	testCases := []struct {
		Kind             string
		APIGroupVersions []schema.GroupVersion
		DefaultVersions  []schema.GroupVersion

		Resource             schema.GroupVersionResource
		ExpectedGroupVersion *schema.GroupVersion
		Err                  bool
	}{
		{Kind: "Unknown", Err: true},
		{Kind: "InternalObject", Err: true},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "Unknown", Err: true},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: testGroupVersion.WithResource("internalobjects")},
		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: testGroupVersion.WithResource("internalobjects")},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: testGroupVersion.WithResource("internalobjects")},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{}, Resource: internalGroupVersion.WithResource("internalobjects"), ExpectedGroupVersion: &schema.GroupVersion{Group: testGroup, Version: "test"}},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{{Group: testGroup, Version: "test"}}, Resource: testGroupVersion.WithResource("internalobjects")},

		// TODO: add test for a resource that exists in one version but not another
	}
	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.DefaultVersions)
		mapper.Add(internalGroupVersion.WithKind("InternalObject"), RESTScopeNamespace)

		preferredVersions := []string{}
		for _, gv := range testCase.APIGroupVersions {
			preferredVersions = append(preferredVersions, gv.Version)
		}
		gk := schema.GroupKind{Group: testGroup, Kind: testCase.Kind}

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
	expectedGroupVersion1 := schema.GroupVersion{Group: "tgroup", Version: "test1"}
	expectedGroupVersion2 := schema.GroupVersion{Group: "tgroup", Version: "test2"}
	expectedGroupVersion3 := schema.GroupVersion{Group: "tgroup", Version: "test3"}
	internalObjectGK := schema.GroupKind{Group: "tgroup", Kind: "InternalObject"}
	otherObjectGK := schema.GroupKind{Group: "tgroup", Kind: "OtherObject"}

	mapper := NewDefaultRESTMapper([]schema.GroupVersion{expectedGroupVersion1, expectedGroupVersion2})
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace)
	mapper.Add(expectedGroupVersion2.WithKind("OtherObject"), RESTScopeNamespace)

	// pick default matching object kind based on search order
	mapping, err := mapper.RESTMapping(otherObjectGK)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != expectedGroupVersion2.WithResource("otherobjects") || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion2 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}

	mapping, err = mapper.RESTMapping(internalObjectGK)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != expectedGroupVersion1.WithResource("internalobjects") || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion1 {
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
	if mapping.Resource != expectedGroupVersion2.WithResource("otherobjects") || mapping.GroupVersionKind.GroupVersion() != expectedGroupVersion2 {
		t.Errorf("unexpected mapping: %#v", mapping)
	}
}

func TestRESTMapperRESTMappings(t *testing.T) {
	testGroup := "tgroup"
	testGroupVersion := schema.GroupVersion{Group: testGroup, Version: "v1"}

	testCases := []struct {
		Kind                string
		APIGroupVersions    []schema.GroupVersion
		DefaultVersions     []schema.GroupVersion
		AddGroupVersionKind []schema.GroupVersionKind

		ExpectedRESTMappings []*RESTMapping
		Err                  bool
	}{
		{Kind: "Unknown", Err: true},
		{Kind: "InternalObject", Err: true},

		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "Unknown", Err: true},

		// ask for specific version - not available - thus error
		{DefaultVersions: []schema.GroupVersion{testGroupVersion}, Kind: "InternalObject", APIGroupVersions: []schema.GroupVersion{{Group: testGroup, Version: "v2"}}, Err: true},

		// ask for specific version - available - check ExpectedRESTMappings
		{
			DefaultVersions:      []schema.GroupVersion{testGroupVersion},
			Kind:                 "InternalObject",
			APIGroupVersions:     []schema.GroupVersion{{Group: testGroup, Version: "v2"}},
			AddGroupVersionKind:  []schema.GroupVersionKind{schema.GroupVersion{Group: testGroup, Version: "v2"}.WithKind("InternalObject")},
			ExpectedRESTMappings: []*RESTMapping{{Resource: schema.GroupVersionResource{Group: testGroup, Version: "v2", Resource: "internalobjects"}, GroupVersionKind: schema.GroupVersionKind{Group: testGroup, Version: "v2", Kind: "InternalObject"}}},
		},

		// ask for specific versions - only one available - check ExpectedRESTMappings
		{
			DefaultVersions:      []schema.GroupVersion{testGroupVersion},
			Kind:                 "InternalObject",
			APIGroupVersions:     []schema.GroupVersion{{Group: testGroup, Version: "v3"}, {Group: testGroup, Version: "v2"}},
			AddGroupVersionKind:  []schema.GroupVersionKind{schema.GroupVersion{Group: testGroup, Version: "v2"}.WithKind("InternalObject")},
			ExpectedRESTMappings: []*RESTMapping{{Resource: schema.GroupVersionResource{Group: testGroup, Version: "v2", Resource: "internalobjects"}, GroupVersionKind: schema.GroupVersionKind{Group: testGroup, Version: "v2", Kind: "InternalObject"}}},
		},

		// do not ask for specific version - search through default versions - check ExpectedRESTMappings
		{
			DefaultVersions:     []schema.GroupVersion{testGroupVersion, {Group: testGroup, Version: "v2"}},
			Kind:                "InternalObject",
			AddGroupVersionKind: []schema.GroupVersionKind{schema.GroupVersion{Group: testGroup, Version: "v1"}.WithKind("InternalObject"), schema.GroupVersion{Group: testGroup, Version: "v2"}.WithKind("InternalObject")},
			ExpectedRESTMappings: []*RESTMapping{
				{
					Resource:         schema.GroupVersionResource{Group: testGroup, Version: "v1", Resource: "internalobjects"},
					GroupVersionKind: schema.GroupVersionKind{Group: testGroup, Version: "v1", Kind: "InternalObject"},
				},
				{
					Resource:         schema.GroupVersionResource{Group: testGroup, Version: "v2", Resource: "internalobjects"},
					GroupVersionKind: schema.GroupVersionKind{Group: testGroup, Version: "v2", Kind: "InternalObject"},
				},
			},
		},
	}

	for i, testCase := range testCases {
		mapper := NewDefaultRESTMapper(testCase.DefaultVersions)
		for _, gvk := range testCase.AddGroupVersionKind {
			mapper.Add(gvk, RESTScopeNamespace)
		}

		preferredVersions := []string{}
		for _, gv := range testCase.APIGroupVersions {
			preferredVersions = append(preferredVersions, gv.Version)
		}
		gk := schema.GroupKind{Group: testGroup, Kind: testCase.Kind}

		mappings, err := mapper.RESTMappings(gk, preferredVersions...)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: unexpected error behavior %t: %v", i, testCase.Err, err)
		}
		if hasErr {
			continue
		}
		if len(mappings) != len(testCase.ExpectedRESTMappings) {
			t.Errorf("%d: unexpected number = %d of rest mappings was returned, expected = %d", i, len(mappings), len(testCase.ExpectedRESTMappings))
		}
		for j, mapping := range mappings {
			exp := testCase.ExpectedRESTMappings[j]
			if mapping.Resource != exp.Resource {
				t.Errorf("%d - %d: unexpected resource: %#v", i, j, mapping)
			}
			if mapping.GroupVersionKind != exp.GroupVersionKind {
				t.Errorf("%d - %d: unexpected GroupVersionKind: %#v", i, j, mapping)
			}
		}
	}
}

func TestRESTMapperReportsErrorOnBadVersion(t *testing.T) {
	expectedGroupVersion1 := schema.GroupVersion{Group: "tgroup", Version: "test1"}
	expectedGroupVersion2 := schema.GroupVersion{Group: "tgroup", Version: "test2"}
	internalObjectGK := schema.GroupKind{Group: "tgroup", Kind: "InternalObject"}

	mapper := NewDefaultRESTMapper([]schema.GroupVersion{expectedGroupVersion1, expectedGroupVersion2})
	mapper.Add(expectedGroupVersion1.WithKind("InternalObject"), RESTScopeNamespace)
	_, err := mapper.RESTMapping(internalObjectGK, "test3")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
