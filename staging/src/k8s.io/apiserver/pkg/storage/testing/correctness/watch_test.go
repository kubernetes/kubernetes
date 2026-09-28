/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// TestValidateWatch pairs a fixed event history with a watch stream and asserts
// whether ValidateWatch accepts it. Each rejected case breaks one of the
// guarantees a watch is supposed to provide.
func TestValidateWatch(t *testing.T) {
	pod1 := newTestPod("pod1", "ns1", "uid-1", "")
	pod2 := newTestPod("pod2", "ns1", "uid-2", "")
	pod3 := newTestPod("pod3", "ns1", "uid-3", "")

	var (
		addPod1RV2    = watch.Event{Type: watch.Added, Object: withRV(pod1, "2")}
		addPod2RV3    = watch.Event{Type: watch.Added, Object: withRV(pod2, "3")}
		bluePod1RV4   = watch.Event{Type: watch.Modified, Object: withLabel(pod1, "4", "color", "blue")}
		deletePod2RV5 = watch.Event{Type: watch.Deleted, Object: withRV(pod2, "5")}
		redPod1RV6    = watch.Event{Type: watch.Modified, Object: withLabel(pod1, "6", "color", "red")}
		pod1RV7       = watch.Event{Type: watch.Modified, Object: dropLabel(pod1, "7", "color")}
		deletePod1RV8 = watch.Event{Type: watch.Deleted, Object: withRV(pod1, "8")}
	)
	history := []Change{
		{ResourceVersion: 2, Object: addPod1RV2.Object},
		{ResourceVersion: 3, Object: addPod2RV3.Object},
		{ResourceVersion: 4, Object: bluePod1RV4.Object, PrevObject: addPod1RV2.Object},
		{ResourceVersion: 5, PrevObject: addPod2RV3.Object},
		{ResourceVersion: 6, Object: redPod1RV6.Object, PrevObject: bluePod1RV4.Object},
		{ResourceVersion: 7, Object: pod1RV7.Object, PrevObject: redPod1RV6.Object},
		{ResourceVersion: 8, PrevObject: pod1RV7.Object},
	}
	// Events for object transitions between selectors.
	var (
		addBluePod1RV4    = watch.Event{Type: watch.Added, Object: bluePod1RV4.Object}
		deleteBluePod1RV6 = watch.Event{Type: watch.Deleted, Object: withLabel(pod1, "6", "color", "blue")}
		addRedPod1RV6     = watch.Event{Type: watch.Added, Object: redPod1RV6.Object}
		deleteRedPod1RV7  = watch.Event{Type: watch.Deleted, Object: withLabel(pod1, "7", "color", "red")}
	)
	isBlue := storage.SelectionPredicate{
		Label:    labels.SelectorFromSet(labels.Set{"color": "blue"}),
		Field:    fields.Everything(),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isRed := storage.SelectionPredicate{
		Label:    labels.SelectorFromSet(labels.Set{"color": "red"}),
		Field:    fields.Everything(),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isPod1 := storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.OneTermEqualSelector("metadata.name", "pod1"),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isPod2 := storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.OneTermEqualSelector("metadata.name", "pod2"),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}

	tests := []struct {
		name        string
		requests    []WatchRequest
		events      []watch.Event
		expectError bool
	}{
		{
			name: "whole history",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "partial history from beginning",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
			},
			events: []watch.Event{addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "partial history from the end",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4},
		},
		{
			name: "watch opened on exact resource version",
			requests: []WatchRequest{
				{ResourceVersion: "3"},
			},
			events: []watch.Event{bluePod1RV4, deletePod2RV5},
		},
		{
			name: "missing event in the middle",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod2RV3 /*bluePod1RV4,*/, deletePod2RV5},
			expectError: true,
		},
		{
			name: "duplicate event",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, addPod2RV3, bluePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of different type",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, deletePod2RV5, bluePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of the same type",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod2RV3, addPod1RV2, bluePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "event the history never produced",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{
				addPod1RV2,
				{Type: watch.Added, Object: withRV(pod3, "3")},
				addPod2RV3, bluePod1RV4, deletePod2RV5,
			},
			expectError: true,
		},
		{
			name: "wrong event type",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{
				addPod1RV2,
				{Type: watch.Modified, Object: withRV(pod2, "3")},
				bluePod1RV4, deletePod2RV5,
			},
			expectError: true,
		},
		{
			name: "wrong object in event",
			requests: []WatchRequest{
				{ResourceVersion: "1"},
			},
			events: []watch.Event{
				addPod1RV2,
				{Type: watch.Added, Object: withRV(pod3, "2")},
				bluePod1RV4, deletePod2RV5,
			},
			expectError: true,
		},
		{
			name: "event at the requested resource version",
			requests: []WatchRequest{
				{ResourceVersion: "3"},
			},
			events:      []watch.Event{addPod2RV3, bluePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "bookmark after every event events",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{newBookmark("1"), addPod1RV2, newBookmark("2"), addPod2RV3, newBookmark("3"), bluePod1RV4, newBookmark("4"), deletePod2RV5, newBookmark("5")},
		},
		{
			name: "bookmark after the event",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
			},
			events: []watch.Event{
				addPod2RV3,
				newBookmark("3"),
			},
		},
		{
			name: "bookmark ahead of the event",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
			},
			events: []watch.Event{
				newBookmark("3"),
				addPod2RV3,
			},
			expectError: true,
		},
		{
			name: "bookmark on fresh watch",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "2"},
			},
			events: []watch.Event{
				newBookmark("3"),
			},
		},
		{
			name: "bookmark on fresh watch",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "2"},
			},
			events: []watch.Event{
				addPod2RV3,
				newBookmark("3"),
				newBookmark("3"),
			},
		},
		{
			name: "label selector color=blue",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events: []watch.Event{addBluePod1RV4, deleteBluePod1RV6},
		},
		{
			name: "label selector opened on exact resource version",
			requests: []WatchRequest{
				{ResourceVersion: "4", Predicate: isBlue},
				{ResourceVersion: "5", Predicate: isBlue},
			},
			events: []watch.Event{deleteBluePod1RV6},
		},
		{
			name: "label selector color=red",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isRed},
				{ResourceVersion: "0", Predicate: isRed},
				{ResourceVersion: "1", Predicate: isRed},
				{ResourceVersion: "2", Predicate: isRed},
				{ResourceVersion: "3", Predicate: isRed},
				{ResourceVersion: "4", Predicate: isRed},
				{ResourceVersion: "5", Predicate: isRed},
			},
			events: []watch.Event{addRedPod1RV6, deleteRedPod1RV7},
		},
		{
			name: "field selector name=pod1",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isPod1},
				{ResourceVersion: "0", Predicate: isPod1},
				{ResourceVersion: "1", Predicate: isPod1},
			},
			events: []watch.Event{addPod1RV2, bluePod1RV4, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "field selector name=pod2",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isPod2},
				{ResourceVersion: "0", Predicate: isPod2},
				{ResourceVersion: "1", Predicate: isPod2},
				{ResourceVersion: "2", Predicate: isPod2},
			},
			events: []watch.Event{addPod2RV3, deletePod2RV5},
		},
		{
			name: "event not matching the selector",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
			},
			events:      []watch.Event{addPod1RV2, addBluePod1RV4, deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "missing move into the selector",
			requests: []WatchRequest{
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events:      []watch.Event{ /*addBluePod1RV4,*/ deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "move into the selector as modified instead of add",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events:      []watch.Event{bluePod1RV4, deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "move out of the selector as modified instead of delete",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events:      []watch.Event{addBluePod1RV4, redPod1RV6},
			expectError: true,
		},
		{
			name: "move out of the selector with update object instead of previous object",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events: []watch.Event{
				addBluePod1RV4,
				{Type: watch.Deleted, Object: redPod1RV6.Object},
			},
			expectError: true,
		},
		{
			name: "move out of the selector with update object instead of previous object",
			requests: []WatchRequest{
				{ResourceVersion: "", Predicate: isBlue},
				{ResourceVersion: "0", Predicate: isBlue},
				{ResourceVersion: "1", Predicate: isBlue},
				{ResourceVersion: "2", Predicate: isBlue},
				{ResourceVersion: "3", Predicate: isBlue},
			},
			events: []watch.Event{
				addBluePod1RV4,
				{Type: watch.Deleted, Object: bluePod1RV4.Object},
			},
			expectError: true,
		},
	}

	validator := NewWatchValidator(storage.APIObjectVersioner{}, getKey, history)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, req := range tc.requests {
				err := validator.ValidateWatch(req, WatchResponse{Events: tc.events})
				if !tc.expectError {
					require.NoError(t, err, "%+v: unexpected error %v", req, err)
					continue
				}
				require.Error(t, err, "%+v: expected error", req)
			}
		})
	}
}

func newBookmark(rv string) watch.Event {
	return watch.Event{Type: watch.Bookmark, Object: newTestPod("", "", "", rv)}
}
