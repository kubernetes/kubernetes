/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestPriorityRESTMapperResourceForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		delegate         RESTMapper
		resourcePatterns []schema.GroupVersionResource
		result           schema.GroupVersionResource
		err              string
	}{
		{
			name:     "error",
			delegate: fixedRESTMapper{err: errors.New("delegateError")},
			err:      "delegateError",
		},
		{
			name:     "single hit + error",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "single-hit"}}, err: errors.New("delegateError")},
			result:   schema.GroupVersionResource{Resource: "single-hit"},
			err:      "delegateError",
		},
		{
			name: "group selection + error",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}, err: errors.New("delegateError")},
			resourcePatterns: []schema.GroupVersionResource{
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
			},
			result: schema.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
			err:    "delegateError",
		},

		{
			name:     "single hit",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{{Resource: "single-hit"}}},
			result:   schema.GroupVersionResource{Resource: "single-hit"},
		},
		{
			name: "ambiguous match",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			err: "matches multiple resources",
		},
		{
			name: "group selection",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			resourcePatterns: []schema.GroupVersionResource{
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
			},
			result: schema.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "empty match continues",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			resourcePatterns: []schema.GroupVersionResource{
				{Group: "fail", Version: AnyVersion, Resource: AnyResource},
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
			},
			result: schema.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "group followed by version selection",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
				{Group: "one", Version: "c", Resource: "third"},
			}},
			resourcePatterns: []schema.GroupVersionResource{
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
				{Group: AnyGroup, Version: "a", Resource: AnyResource},
			},
			result: schema.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "resource selection",
			delegate: fixedRESTMapper{resourcesFor: []schema.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "one", Version: "a", Resource: "second"},
			}},
			resourcePatterns: []schema.GroupVersionResource{
				{Group: AnyGroup, Version: AnyVersion, Resource: "second"},
			},
			result: schema.GroupVersionResource{Group: "one", Version: "a", Resource: "second"},
		},
	}

	for _, tc := range tcs {
		mapper := PriorityRESTMapper{Delegate: tc.delegate, ResourcePriority: tc.resourcePatterns}

		actualResult, actualErr := mapper.ResourceFor(schema.GroupVersionResource{})
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if len(tc.err) == 0 && actualErr == nil {
			continue
		}
		if len(tc.err) == 0 && actualErr != nil {
			t.Errorf("%s: unexpected err: %v", tc.name, actualErr)
			continue
		}
		if len(tc.err) > 0 && actualErr == nil {
			t.Errorf("%s: missing expected err: %v", tc.name, tc.err)
			continue
		}
		if !strings.Contains(actualErr.Error(), tc.err) {
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestPriorityRESTMapperKindForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		delegate     RESTMapper
		kindPatterns []schema.GroupVersionKind
		result       schema.GroupVersionKind
		err          string
	}{
		{
			name:     "error",
			delegate: fixedRESTMapper{err: errors.New("delegateErr")},
			err:      "delegateErr",
		},
		{
			name:     "single hit + error",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "single-hit"}}, err: errors.New("delegateErr")},
			result:   schema.GroupVersionKind{Kind: "single-hit"},
			err:      "delegateErr",
		},
		{
			name: "group selection + error",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}, err: errors.New("delegateErr")},
			kindPatterns: []schema.GroupVersionKind{
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
			},
			result: schema.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
			err:    "delegateErr",
		},

		{
			name:     "single hit",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{{Kind: "single-hit"}}},
			result:   schema.GroupVersionKind{Kind: "single-hit"},
		},
		{
			name: "ambiguous match",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			err: "matches multiple kinds",
		},
		{
			name: "group selection",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			kindPatterns: []schema.GroupVersionKind{
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
			},
			result: schema.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "empty match continues",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			kindPatterns: []schema.GroupVersionKind{
				{Group: "fail", Version: AnyVersion, Kind: AnyKind},
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
			},
			result: schema.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "group followed by version selection",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
				{Group: "one", Version: "c", Kind: "third"},
			}},
			kindPatterns: []schema.GroupVersionKind{
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
				{Group: AnyGroup, Version: "a", Kind: AnyKind},
			},
			result: schema.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "kind selection",
			delegate: fixedRESTMapper{kindsFor: []schema.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "one", Version: "a", Kind: "second"},
			}},
			kindPatterns: []schema.GroupVersionKind{
				{Group: AnyGroup, Version: AnyVersion, Kind: "second"},
			},
			result: schema.GroupVersionKind{Group: "one", Version: "a", Kind: "second"},
		},
	}

	for _, tc := range tcs {
		mapper := PriorityRESTMapper{Delegate: tc.delegate, KindPriority: tc.kindPatterns}

		actualResult, actualErr := mapper.KindFor(schema.GroupVersionResource{})
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if len(tc.err) == 0 && actualErr == nil {
			continue
		}
		if len(tc.err) == 0 && actualErr != nil {
			t.Errorf("%s: unexpected err: %v", tc.name, actualErr)
			continue
		}
		if len(tc.err) > 0 && actualErr == nil {
			t.Errorf("%s: missing expected err: %v", tc.name, tc.err)
			continue
		}
		if !strings.Contains(actualErr.Error(), tc.err) {
			t.Errorf("%s: expected %v, got %v", tc.name, tc.err, actualErr)
		}
	}
}

func TestPriorityRESTMapperRESTMapping(t *testing.T) {
	mapping1 := &RESTMapping{
		GroupVersionKind: schema.GroupVersionKind{Kind: "Foo", Version: "v1alpha1"},
	}
	mapping2 := &RESTMapping{
		GroupVersionKind: schema.GroupVersionKind{Kind: "Foo", Version: "v1"},
	}
	mapping3 := &RESTMapping{
		GroupVersionKind: schema.GroupVersionKind{Group: "other", Kind: "Foo", Version: "v1"},
	}
	allMappers := MultiRESTMapper{
		fixedRESTMapper{mappings: []*RESTMapping{mapping1}},
		fixedRESTMapper{mappings: []*RESTMapping{mapping2}},
		fixedRESTMapper{mappings: []*RESTMapping{mapping3}},
	}
	tcs := []struct {
		name string

		mapper PriorityRESTMapper
		input  schema.GroupKind
		result *RESTMapping
		err    error
	}{
		{
			name:   "empty",
			mapper: PriorityRESTMapper{Delegate: MultiRESTMapper{}},
			input:  schema.GroupKind{Kind: "Foo"},
			err:    &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}},
		},
		{
			name:   "ignore not found",
			mapper: PriorityRESTMapper{Delegate: MultiRESTMapper{fixedRESTMapper{err: &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "IGNORE_THIS"}}}}},
			input:  schema.GroupKind{Kind: "Foo"},
			err:    &NoKindMatchError{GroupKind: schema.GroupKind{Kind: "Foo"}},
		},
		{
			name:   "accept first failure",
			mapper: PriorityRESTMapper{Delegate: MultiRESTMapper{fixedRESTMapper{err: errors.New("fail on this")}, fixedRESTMapper{mappings: []*RESTMapping{mapping1}}}},
			input:  schema.GroupKind{Kind: "Foo"},
			err:    errors.New("fail on this"),
		},
		{
			name:   "result + error",
			mapper: PriorityRESTMapper{Delegate: fixedRESTMapper{mappings: []*RESTMapping{mapping1}, err: errors.New("fail on this")}},
			input:  schema.GroupKind{Kind: "Foo"},
			result: mapping1,
			err:    errors.New("fail on this"),
		},
		{
			name: "return error for ambiguous",
			mapper: PriorityRESTMapper{
				Delegate: allMappers,
			},
			input: schema.GroupKind{Kind: "Foo"},
			err: &AmbiguousKindError{
				PartialKind: schema.GroupVersionKind{Kind: "Foo"},
				MatchingKinds: []schema.GroupVersionKind{
					{Kind: "Foo", Version: "v1alpha1"},
					{Kind: "Foo", Version: "v1"},
					{Group: "other", Kind: "Foo", Version: "v1"},
				},
			},
		},
		{
			name: "accept only item",
			mapper: PriorityRESTMapper{
				Delegate: fixedRESTMapper{mappings: []*RESTMapping{mapping1}},
			},
			input:  schema.GroupKind{Kind: "Foo"},
			result: mapping1,
		},
		{
			name: "return single priority",
			mapper: PriorityRESTMapper{
				Delegate:     allMappers,
				KindPriority: []schema.GroupVersionKind{{Version: "v1", Kind: AnyKind}, {Version: "v1alpha1", Kind: AnyKind}},
			},
			input:  schema.GroupKind{Kind: "Foo"},
			result: mapping2,
		},
		{
			name: "return out of group match",
			mapper: PriorityRESTMapper{
				Delegate:     allMappers,
				KindPriority: []schema.GroupVersionKind{{Group: AnyGroup, Version: "v1", Kind: AnyKind}, {Group: "other", Version: AnyVersion, Kind: AnyKind}},
			},
			input:  schema.GroupKind{Kind: "Foo"},
			result: mapping3,
		},
	}

	for _, tc := range tcs {
		actualResult, actualErr := tc.mapper.RESTMapping(tc.input)
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

func TestPriorityRESTMapperRESTMappingHonorsUserVersion(t *testing.T) {
	mappingV2alpha1 := &RESTMapping{
		GroupVersionKind: schema.GroupVersionKind{Group: "Bar", Kind: "Foo", Version: "v2alpha1"},
	}
	mappingV1 := &RESTMapping{
		GroupVersionKind: schema.GroupVersionKind{Group: "Bar", Kind: "Foo", Version: "v1"},
	}

	allMappers := MultiRESTMapper{
		fixedRESTMapper{mappings: []*RESTMapping{mappingV2alpha1}},
		fixedRESTMapper{mappings: []*RESTMapping{mappingV1}},
	}

	mapper := PriorityRESTMapper{
		Delegate:     allMappers,
		KindPriority: []schema.GroupVersionKind{{Group: "Bar", Version: "v2alpha1", Kind: AnyKind}, {Group: "Bar", Version: AnyVersion, Kind: AnyKind}},
	}

	outMapping1, err := mapper.RESTMapping(schema.GroupKind{Group: "Bar", Kind: "Foo"}, "v1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if outMapping1 != mappingV1 {
		t.Errorf("asked for version %v, expected mapping for %v, got mapping for %v", "v1", mappingV1.GroupVersionKind, outMapping1.GroupVersionKind)
	}

	outMapping2, err := mapper.RESTMapping(schema.GroupKind{Group: "Bar", Kind: "Foo"}, "v2alpha1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if outMapping2 != mappingV2alpha1 {
		t.Errorf("asked for version %v, expected mapping for %v, got mapping for %v", "v2alpha1", mappingV2alpha1.GroupVersionKind, outMapping2.GroupVersionKind)
	}
}
