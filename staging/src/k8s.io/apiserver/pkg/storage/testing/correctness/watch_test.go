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

	// Four writes over two keys, covering add, modify and delete.
	var (
		addPod1RV2    = watch.Event{Type: watch.Added, Object: withRV(pod1, "2")}
		addPod2RV3    = watch.Event{Type: watch.Added, Object: withRV(pod2, "3")}
		updatePod1RV4 = watch.Event{Type: watch.Modified, Object: withRV(pod1, "4")}
		deletePod2RV5 = watch.Event{Type: watch.Deleted, Object: withRV(pod2, "5")}
	)
	history := []watch.Event{addPod1RV2, addPod2RV3, updatePod1RV4, deletePod2RV5}

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
			events: []watch.Event{addPod1RV2, addPod2RV3, updatePod1RV4, deletePod2RV5},
		},
		{
			name: "partial history from beginning",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
			},
			events: []watch.Event{addPod2RV3, updatePod1RV4, deletePod2RV5},
		},
		{
			name: "partial history from the end",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, updatePod1RV4},
		},
		{
			name: "watch opened on exact resource version",
			requests: []WatchRequest{
				{ResourceVersion: "3"},
			},
			events: []watch.Event{updatePod1RV4, deletePod2RV5},
		},
		{
			name: "missing event in the middle",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod2RV3 /*updatePod1RV4,*/, deletePod2RV5},
			expectError: true,
		},
		{
			name: "duplicate event",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, addPod2RV3, updatePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of different type",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, deletePod2RV5, updatePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of the same type",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events:      []watch.Event{addPod2RV3, addPod1RV2, updatePod1RV4, deletePod2RV5},
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
				addPod2RV3, updatePod1RV4, deletePod2RV5,
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
				updatePod1RV4, deletePod2RV5,
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
				updatePod1RV4, deletePod2RV5,
			},
			expectError: true,
		},
		{
			name: "event at the requested resource version",
			requests: []WatchRequest{
				{ResourceVersion: "3"},
			},
			events:      []watch.Event{addPod2RV3, updatePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "bookmark after every event events",
			requests: []WatchRequest{
				{ResourceVersion: ""},
				{ResourceVersion: "0"},
				{ResourceVersion: "1"},
			},
			events: []watch.Event{newBookmark("1"), addPod1RV2, newBookmark("2"), addPod2RV3, newBookmark("3"), updatePod1RV4, newBookmark("4"), deletePod2RV5, newBookmark("5")},
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
