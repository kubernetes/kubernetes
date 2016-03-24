/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestPriorityRESTMapperResourceForErrorHandling(t *testing.T) {
	tcs := []struct {
		name string

		delegate         RESTMapper
		resourcePatterns []unversioned.GroupVersionResource
		result           unversioned.GroupVersionResource
		err              string
	}{
		{
			name:     "single hit",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{{Resource: "single-hit"}}},
			result:   unversioned.GroupVersionResource{Resource: "single-hit"},
		},
		{
			name: "ambiguous match",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			err: "matches multiple resources",
		},
		{
			name: "group selection",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			resourcePatterns: []unversioned.GroupVersionResource{
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
			},
			result: unversioned.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "empty match continues",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
			}},
			resourcePatterns: []unversioned.GroupVersionResource{
				{Group: "fail", Version: AnyVersion, Resource: AnyResource},
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
			},
			result: unversioned.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "group followed by version selection",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "two", Version: "b", Resource: "second"},
				{Group: "one", Version: "c", Resource: "third"},
			}},
			resourcePatterns: []unversioned.GroupVersionResource{
				{Group: "one", Version: AnyVersion, Resource: AnyResource},
				{Group: AnyGroup, Version: "a", Resource: AnyResource},
			},
			result: unversioned.GroupVersionResource{Group: "one", Version: "a", Resource: "first"},
		},
		{
			name: "resource selection",
			delegate: fixedRESTMapper{resourcesFor: []unversioned.GroupVersionResource{
				{Group: "one", Version: "a", Resource: "first"},
				{Group: "one", Version: "a", Resource: "second"},
			}},
			resourcePatterns: []unversioned.GroupVersionResource{
				{Group: AnyGroup, Version: AnyVersion, Resource: "second"},
			},
			result: unversioned.GroupVersionResource{Group: "one", Version: "a", Resource: "second"},
		},
	}

	for _, tc := range tcs {
		mapper := PriorityRESTMapper{Delegate: tc.delegate, ResourcePriority: tc.resourcePatterns}

		actualResult, actualErr := mapper.ResourceFor(unversioned.GroupVersionResource{})
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if len(tc.err) == 0 && actualErr == nil {
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
		kindPatterns []unversioned.GroupVersionKind
		result       unversioned.GroupVersionKind
		err          string
	}{
		{
			name:     "single hit",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{{Kind: "single-hit"}}},
			result:   unversioned.GroupVersionKind{Kind: "single-hit"},
		},
		{
			name: "ambiguous match",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			err: "matches multiple kinds",
		},
		{
			name: "group selection",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			kindPatterns: []unversioned.GroupVersionKind{
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
			},
			result: unversioned.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "empty match continues",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
			}},
			kindPatterns: []unversioned.GroupVersionKind{
				{Group: "fail", Version: AnyVersion, Kind: AnyKind},
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
			},
			result: unversioned.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "group followed by version selection",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "two", Version: "b", Kind: "second"},
				{Group: "one", Version: "c", Kind: "third"},
			}},
			kindPatterns: []unversioned.GroupVersionKind{
				{Group: "one", Version: AnyVersion, Kind: AnyKind},
				{Group: AnyGroup, Version: "a", Kind: AnyKind},
			},
			result: unversioned.GroupVersionKind{Group: "one", Version: "a", Kind: "first"},
		},
		{
			name: "kind selection",
			delegate: fixedRESTMapper{kindsFor: []unversioned.GroupVersionKind{
				{Group: "one", Version: "a", Kind: "first"},
				{Group: "one", Version: "a", Kind: "second"},
			}},
			kindPatterns: []unversioned.GroupVersionKind{
				{Group: AnyGroup, Version: AnyVersion, Kind: "second"},
			},
			result: unversioned.GroupVersionKind{Group: "one", Version: "a", Kind: "second"},
		},
	}

	for _, tc := range tcs {
		mapper := PriorityRESTMapper{Delegate: tc.delegate, KindPriority: tc.kindPatterns}

		actualResult, actualErr := mapper.KindFor(unversioned.GroupVersionResource{})
		if e, a := tc.result, actualResult; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
		if len(tc.err) == 0 && actualErr == nil {
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
