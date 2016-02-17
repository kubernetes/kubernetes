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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestMultiRESTMapperResourceForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  unversioned.GroupVersionResource
		result unversioned.GroupVersionResource
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionResource{},
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionResource{},
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{resourceFor: unversioned.GroupVersionResource{Resource: "unused"}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionResource{},
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.ResourceFor(tc.input)
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.err.Error(), actualErr.Error(); e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}

func TestMultiRESTMapperResourcesForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  unversioned.GroupVersionResource
		result []unversioned.GroupVersionResource
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{{Resource: "unused"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.ResourcesFor(tc.input)
		if e, a := tc.result, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.err.Error(), actualErr.Error(); e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}

func TestMultiRESTMapperKindsForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  unversioned.GroupVersionResource
		result []unversioned.GroupVersionKind
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{{Kind: "unused"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: nil,
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.KindsFor(tc.input)
		if e, a := tc.result, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.err.Error(), actualErr.Error(); e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}

func TestMultiRESTMapperKindForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		mapper MultiRESTMapper
		input  unversioned.GroupVersionResource
		result unversioned.GroupVersionKind
		err    error
	}{
		{
			name:   "empty",
			mapper: MultiRESTMapper{},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionKind{},
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "ignore not found",
			mapper: MultiRESTMapper{fixedRESTMapper{err: &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "IGNORE_THIS"}}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionKind{},
			err:    &NoResourceMatchError{PartialResource: unversioned.GroupVersionResource{Resource: "foo"}},
		},
		{
			name:   "accept first failure",
			mapper: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{kindFor: unversioned.GroupVersionKind{Kind: "unused"}}},
			input:  unversioned.GroupVersionResource{Resource: "foo"},
			result: unversioned.GroupVersionKind{},
			err:    errors.New("fail on this"),
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.KindFor(tc.input)
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.err.Error(), actualErr.Error(); e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}

type fixedRESTMapper struct {
	resourcesFor []unversioned.GroupVersionResource
	kindsFor     []unversioned.GroupVersionKind
	resourceFor  unversioned.GroupVersionResource
	kindFor      unversioned.GroupVersionKind

	err error
}

func (m fixedRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return "", m.err
}

func (m fixedRESTMapper) ResourcesFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	return m.resourcesFor, m.err
}

func (m fixedRESTMapper) KindsFor(resource unversioned.GroupVersionResource) (gvk []unversioned.GroupVersionKind, err error) {
	return m.kindsFor, m.err
}

func (m fixedRESTMapper) ResourceFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	return m.resourceFor, m.err
}

func (m fixedRESTMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	return m.kindFor, m.err
}

func (m fixedRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	return nil, m.err
}

func (m fixedRESTMapper) AliasesForResource(alias string) (aliases []string, ok bool) {
	return nil, false
}

func (m fixedRESTMapper) ResourceIsValid(resource unversioned.GroupVersionResource) bool {
	return false
}
