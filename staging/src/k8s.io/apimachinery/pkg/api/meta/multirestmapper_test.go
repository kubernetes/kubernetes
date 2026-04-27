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
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestMultiRESTMapperResourceFor(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  schema.GroupVersionResource
		result schema.GroupVersionResource
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionResource{},
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionResource{},
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "unused"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionResource{},
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.ResourceFor(tc.input)
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		switch {
		case tc.err == nil && actualErr == nil:
		case tc.err == nil:
			t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
		case actualErr == nil:
			t.Errorf("%s: expected error: %v got nil", tc.name, tc.err)
		case tc.err.Error() != actualErr.Error():
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestMultiRESTMapperResourcesFor(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  schema.GroupVersionResource
		result []schema.GroupVersionResource
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "unused"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    errors.New("fail on this"),
		},
		{
			name: "union and dedup",
			mapper: MultiRESTMapper{
				fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "dupe"}, {Resource: "first"}}},
				fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "dupe"}, {Resource: "second"}}},
			},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: []schema.GroupVersionResource{{Resource: "dupe"}, {Resource: "first"}, {Resource: "second"}},
		},
		{
			name: "skip not and continue",
			mapper: MultiRESTMapper{
				fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}},
				fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "first"}, {Resource: "second"}}},
			},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: []schema.GroupVersionResource{{Resource: "first"}, {Resource: "second"}},
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.ResourcesFor(tc.input)
		if e, a := tc.result, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		switch {
		case tc.err == nil && actualErr == nil:
		case tc.err == nil:
			t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
		case actualErr == nil:
			t.Errorf("%s: expected error: %v got nil", tc.name, tc.err)
		case tc.err.Error() != actualErr.Error():
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestMultiRESTMapperKindsFor(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  schema.GroupVersionResource
		result []schema.GroupVersionKind
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "unused"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    errors.New("fail on this"),
		},
		{
			name: "union and dedup",
			mapper: MultiRESTMapper{
				fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "dupe"}, {Kind: "first"}}},
				fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "dupe"}, {Kind: "second"}}},
			},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: []schema.GroupVersionKind{{Kind: "dupe"}, {Kind: "first"}, {Kind: "second"}},
		},
		{
			name: "skip not and continue",
			mapper: MultiRESTMapper{
				fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}},
				fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "first"}, {Kind: "second"}}},
			},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: []schema.GroupVersionKind{{Kind: "first"}, {Kind: "second"}},
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.KindsFor(tc.input)
		if e, a := tc.result, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		switch {
		case tc.err == nil && actualErr == nil:
		case tc.err == nil:
			t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
		case actualErr == nil:
			t.Errorf("%s: expected error: %v got nil", tc.name, tc.err)
		case tc.err.Error() != actualErr.Error():
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestMultiRESTMapperKindFor(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  schema.GroupVersionResource
		result schema.GroupVersionKind
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionKind{},
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionKind{},
			err:    &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "unused"}}}},
			input:  schema.GroupVersionResource{Resource: "foo"},
			result: schema.GroupVersionKind{},
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.KindFor(tc.input)
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		switch {
		case tc.err == nil && actualErr == nil:
		case tc.err == nil:
			t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
		case actualErr == nil:
			t.Errorf("%s: expected error: %v got nil", tc.name, tc.err)
		case tc.err.Error() != actualErr.Error():
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestMultiRESTMapperRESTMappings(t *testing.T) {
	mapping1, mapping2 := &RESTMapping{}, &RESTMapping{}
	tcs := []struct {
		name string

		mapper    MultiRESTMapper
		groupKind schema.GroupKind
		versions  []string
		result    []*RESTMapping
		err       error
	}{
		{
			name:      "empty with no versions",
			mapper:    MultiRESTMapper{},
			groupKind: schema.GroupKind{Kind: "Foo"},
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}},
		},
		{
			name:      "empty with one version",
			mapper:    MultiRESTMapper{},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta"},
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1beta"}},
		},
		{
			name:      "empty with multi(two) vesions",
			mapper:    MultiRESTMapper{},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta", "v2"},
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1beta", "v2"}},
		},
		{
			name:      "ignore not found with kind not exist",
			mapper:    MultiRESTMapper{fixedRESTMapper{err: &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "IGNORE_THIS"}}}},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  nil,
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}},
		},
		{
			name:      "ignore not found with version not exist",
			mapper:    MultiRESTMapper{fixedRESTMapper{err: &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1"}}}},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta"},
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1beta"}},
		},
		{
			name:      "ignore not found with multi versions not exist",
			mapper:    MultiRESTMapper{fixedRESTMapper{err: &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1"}}}},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta", "v2"},
			result:    nil,
			err:       &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}, SearchedVersions: []string{"v1beta", "v2"}},
		},
		{
			name:      "accept first failure",
			mapper:    MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{mappings: []*RESTMapping{mapping1}}},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta"},
			result:    nil,
			err:       errors.New("fail on this"),
		},
		{
			name:      "return both",
			mapper:    MultiRESTMapper{fixedRESTMapper{mappings: []*RESTMapping{mapping1}}, fixedRESTMapper{mappings: []*RESTMapping{mapping2}}},
			groupKind: schema.GroupKind{Kind: "Foo"},
			versions:  []string{"v1beta"},
			result:    []*RESTMapping{mapping1, mapping2},
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.RESTMappings(tc.groupKind, tc.versions...)
		if e, a := tc.result, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		switch {
		case tc.err == nil && actualErr == nil:
		case tc.err == nil:
			t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
		case actualErr == nil:
			t.Errorf("%s: expected error: %v got nil", tc.name, tc.err)
		case tc.err.Error() != actualErr.Error():
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

type fixedRESTMapper struct {
	resourcesFor []schema.GroupVersionResource
	kindsFor     []schema.GroupVersionKind
	resourceFor  schema.GroupVersionResource
	kindFor      schema.GroupVersionKind
	mappings     []*RESTMapping

	err error
}

func (m fixedRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return "", m.err
}

func (m fixedRESTMapper) ResourcesFor(resource schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return m.resourcesFor, m.err
}

func (m fixedRESTMapper) KindsFor(resource schema.GroupVersionResource) (gvk []schema.GroupVersionKind, err error) {
	return m.kindsFor, m.err
}

func (m fixedRESTMapper) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	return m.resourceFor, m.err
}

func (m fixedRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	return m.kindFor, m.err
}

func (m fixedRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	return nil, m.err
}

func (m fixedRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) (mappings []*RESTMapping, err error) {
	return m.mappings, m.err
}

func (m fixedRESTMapper) ResourceIsValid(resource schema.GroupVersionResource) bool {
	return false
}
