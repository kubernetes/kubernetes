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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

// TestValidateWatch pairs a fixed event history with a watch stream and asserts
// whether ValidateWatch accepts it. Each rejected case breaks one of the
// guarantees a watch is supposed to provide.
func TestValidateWatch(t *testing.T) {
	pod1 := newTestPod("pod1", "ns1", "uid-1", "")
	pod2 := newTestPod("pod2", "ns1", "uid-2", "")
	pod3 := newTestPod("pod3", "ns1", "uid-3", "")
	pod1Key := mustGetKey(pod1)
	pod2Key := mustGetKey(pod2)

	var (
		addPod1RV2    = watch.Event{Type: watch.Added, Object: withRV(pod1, "2")}
		addPod2RV3    = watch.Event{Type: watch.Added, Object: withRV(pod2, "3")}
		bluePod1RV4   = watch.Event{Type: watch.Modified, Object: withLabel(pod1, "4", "color", "blue")}
		deletePod2RV5 = watch.Event{Type: watch.Deleted, Object: withRV(pod2, "5")}
		redPod1RV6    = watch.Event{Type: watch.Modified, Object: withLabel(pod1, "6", "color", "red")}
		pod1RV7       = watch.Event{Type: watch.Modified, Object: dropLabel(pod1, "7", "color")}
		deletePod1RV8 = watch.Event{Type: watch.Deleted, Object: withRV(pod1, "8")}
	)
	operations := []Operation{
		{
			Request:  Request{Op: OpCreate, Key: pod1Key, Create: CreateRequest{Object: pod1}},
			Response: Response{Object: addPod1RV2.Object},
		},
		{
			Request:  Request{Op: OpCreate, Key: pod2Key, Create: CreateRequest{Object: pod2}},
			Response: Response{Object: addPod2RV3.Object},
		},
		{
			Request:  Request{Op: OpUpdate, Key: pod1Key, Update: UpdateRequest{UpdateFunc: storage.SimpleUpdate(func(runtime.Object) (runtime.Object, error) { return bluePod1RV4.Object, nil })}},
			Response: Response{Object: bluePod1RV4.Object},
		},
		{
			Request:  Request{Op: OpDelete, Key: pod2Key},
			Response: Response{Object: deletePod2RV5.Object},
		},
		{
			Request:  Request{Op: OpUpdate, Key: pod1Key, Update: UpdateRequest{UpdateFunc: storage.SimpleUpdate(func(runtime.Object) (runtime.Object, error) { return redPod1RV6.Object, nil })}},
			Response: Response{Object: redPod1RV6.Object},
		},
		{
			Request:  Request{Op: OpUpdate, Key: pod1Key, Update: UpdateRequest{UpdateFunc: storage.SimpleUpdate(func(runtime.Object) (runtime.Object, error) { return pod1RV7.Object, nil })}},
			Response: Response{Object: pod1RV7.Object},
		},
		{
			Request:  Request{Op: OpDelete, Key: pod1Key},
			Response: Response{Object: deletePod1RV8.Object},
		},
	}
	// Events for object transitions between selectors.
	var (
		addBluePod1RV4    = watch.Event{Type: watch.Added, Object: bluePod1RV4.Object}
		deleteBluePod1RV6 = watch.Event{Type: watch.Deleted, Object: withLabel(pod1, "6", "color", "blue")}
		addRedPod1RV6     = watch.Event{Type: watch.Added, Object: redPod1RV6.Object}
		deleteRedPod1RV7  = watch.Event{Type: watch.Deleted, Object: withLabel(pod1, "7", "color", "red")}
	)
	tests := []struct {
		name        string
		requests    []WatchRequest
		events      []watch.Event
		expectError bool
	}{
		{
			name: "whole history",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "partial history from beginning",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "partial history from the end",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4},
		},
		{
			name: "watch opened on exact resource version",
			requests: []WatchRequest{
				watchEverything("3", ""),
				watchEverything("3", metav1.ResourceVersionMatchExact),
				watchEverything("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{bluePod1RV4, deletePod2RV5},
		},
		{
			name: "watch not older than resource version started later",
			requests: []WatchRequest{
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("2", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{bluePod1RV4, deletePod2RV5},
		},
		{
			name: "watch on exact resource version started later",
			requests: []WatchRequest{
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("2", ""),
				watchEverything("2", metav1.ResourceVersionMatchExact),
			},
			events:      []watch.Event{bluePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "missing event in the middle",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod2RV3 /*bluePod1RV4,*/, deletePod2RV5},
			expectError: true,
		},
		{
			name: "duplicate event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, addPod2RV3, bluePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of different type",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, deletePod2RV5, bluePod1RV4},
			expectError: true,
		},
		{
			name: "events out of order of the same type",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod2RV3, addPod1RV2, bluePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "event the history never produced",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
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
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
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
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
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
				watchEverything("3", ""),
				watchEverything("3", metav1.ResourceVersionMatchExact),
				watchEverything("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod2RV3, bluePod1RV4, deletePod2RV5},
			expectError: true,
		},
		{
			name: "bookmark after every event events",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{newBookmark("1"), addPod1RV2, newBookmark("2"), addPod2RV3, newBookmark("3"), bluePod1RV4, newBookmark("4"), deletePod2RV5, newBookmark("5")},
		},
		{
			name: "bookmark after the event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{
				addPod2RV3,
				newBookmark("3"),
			},
		},
		{
			name: "bookmark ahead of the event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
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
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("2", ""),
				watchEverything("2", metav1.ResourceVersionMatchExact),
				watchEverything("2", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{
				newBookmark("3"),
			},
		},
		{
			name: "bookmark on fresh watch",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("2", ""),
				watchEverything("2", metav1.ResourceVersionMatchExact),
				watchEverything("2", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{
				addPod2RV3,
				newBookmark("3"),
				newBookmark("3"),
			},
		},
		{
			name: "bookmark older than previous event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("2", ""),
				watchEverything("2", metav1.ResourceVersionMatchExact),
				watchEverything("2", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{
				addPod2RV3,
				newBookmark("2"),
			},
			expectError: true,
		},
		{
			name: "label selector color=blue",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addBluePod1RV4, deleteBluePod1RV6},
		},
		{
			name: "label selector opened on exact resource version",
			requests: []WatchRequest{
				watchBlue("4", ""),
				watchBlue("4", metav1.ResourceVersionMatchExact),
				watchBlue("4", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("5", ""),
				watchBlue("5", metav1.ResourceVersionMatchExact),
				watchBlue("5", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{deleteBluePod1RV6},
		},
		{
			name: "label selector color=red",
			requests: []WatchRequest{
				watchRed("", ""),
				watchRed("", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("0", ""),
				watchRed("0", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("1", ""),
				watchRed("1", metav1.ResourceVersionMatchExact),
				watchRed("1", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("2", ""),
				watchRed("2", metav1.ResourceVersionMatchExact),
				watchRed("2", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("3", ""),
				watchRed("3", metav1.ResourceVersionMatchExact),
				watchRed("3", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("4", ""),
				watchRed("4", metav1.ResourceVersionMatchExact),
				watchRed("4", metav1.ResourceVersionMatchNotOlderThan),
				watchRed("5", ""),
				watchRed("5", metav1.ResourceVersionMatchExact),
				watchRed("5", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addRedPod1RV6, deleteRedPod1RV7},
		},
		{
			name: "field selector name=pod1",
			requests: []WatchRequest{
				watchPod1("", ""),
				watchPod1("", metav1.ResourceVersionMatchNotOlderThan),
				watchPod1("0", ""),
				watchPod1("0", metav1.ResourceVersionMatchNotOlderThan),
				watchPod1("1", ""),
				watchPod1("1", metav1.ResourceVersionMatchExact),
				watchPod1("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod1RV2, bluePod1RV4, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "field selector name=pod2",
			requests: []WatchRequest{
				watchPod2("", ""),
				watchPod2("", metav1.ResourceVersionMatchNotOlderThan),
				watchPod2("0", ""),
				watchPod2("0", metav1.ResourceVersionMatchNotOlderThan),
				watchPod2("1", ""),
				watchPod2("1", metav1.ResourceVersionMatchExact),
				watchPod2("1", metav1.ResourceVersionMatchNotOlderThan),
				watchPod2("2", ""),
				watchPod2("2", metav1.ResourceVersionMatchExact),
				watchPod2("2", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod2RV3, deletePod2RV5},
		},
		{
			name: "event not matching the selector",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, addBluePod1RV4, deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "missing move into the selector",
			requests: []WatchRequest{
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
			},
			events:      []watch.Event{ /*addBluePod1RV4,*/ deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "watch not older than resource version started after move into the selector",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{deleteBluePod1RV6},
		},
		{
			name: "move into the selector as modified instead of add",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{bluePod1RV4, deleteBluePod1RV6},
			expectError: true,
		},
		{
			name: "move out of the selector as modified instead of delete",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addBluePod1RV4, redPod1RV6},
			expectError: true,
		},
		{
			name: "move out of the selector with update object instead of previous object",
			requests: []WatchRequest{
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
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
				watchBlue("", ""),
				watchBlue("", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("0", ""),
				watchBlue("0", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("1", ""),
				watchBlue("1", metav1.ResourceVersionMatchExact),
				watchBlue("1", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("2", ""),
				watchBlue("2", metav1.ResourceVersionMatchExact),
				watchBlue("2", metav1.ResourceVersionMatchNotOlderThan),
				watchBlue("3", ""),
				watchBlue("3", metav1.ResourceVersionMatchExact),
				watchBlue("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{
				addBluePod1RV4,
				{Type: watch.Deleted, Object: bluePod1RV4.Object},
			},
			expectError: true,
		},
		{
			name: "watch ending with error event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4, newErrorEvent()},
		},
		{
			name: "watch with only error event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("3", ""),
				watchEverything("3", metav1.ResourceVersionMatchExact),
				watchEverything("3", metav1.ResourceVersionMatchNotOlderThan),
			},
			events: []watch.Event{newErrorEvent()},
		},
		{
			name: "event after error event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, newErrorEvent(), addPod2RV3},
			expectError: true,
		},
		{
			name: "missing event before error event",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2 /*addPod2RV3,*/, bluePod1RV4, newErrorEvent()},
			expectError: true,
		},
		{
			name: "error event with non-status object",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, {Type: watch.Error, Object: withRV(pod2, "3")}},
			expectError: true,
		},
		{
			name: "error event with non-failure status",
			requests: []WatchRequest{
				watchEverything("", ""),
				watchEverything("", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("0", ""),
				watchEverything("0", metav1.ResourceVersionMatchNotOlderThan),
				watchEverything("1", ""),
				watchEverything("1", metav1.ResourceVersionMatchExact),
				watchEverything("1", metav1.ResourceVersionMatchNotOlderThan),
			},
			events:      []watch.Event{addPod1RV2, {Type: watch.Error, Object: &metav1.Status{Status: metav1.StatusSuccess}}},
			expectError: true,
		},
		{
			name: "watchlist from empty state with remaining events",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("0", storage.Everything),
				watchList("1", storage.Everything),
			},
			events: []watch.Event{newInitialEventsEndBookmark("1"), addPod1RV2, addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "watchlist at RV 3 with remaining events",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("0", storage.Everything),
				watchList("1", storage.Everything),
				watchList("2", storage.Everything),
				watchList("3", storage.Everything),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4, deletePod2RV5},
		},
		{
			name: "watchlist at RV 4 with only initial events",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("0", storage.Everything),
				watchList("3", storage.Everything),
				watchList("4", storage.Everything),
			},
			events: []watch.Event{addBluePod1RV4, addPod2RV3, newInitialEventsEndBookmark("4")},
		},
		{
			name: "watchlist with label selector color=blue",
			requests: []WatchRequest{
				watchList("", isBlue),
				watchList("0", isBlue),
				watchList("4", isBlue),
			},
			events: []watch.Event{addBluePod1RV4, newInitialEventsEndBookmark("4"), deleteBluePod1RV6},
		},
		{
			name: "watchlist with only error event",
			requests: []WatchRequest{
				watchList("99", storage.Everything),
			},
			events: []watch.Event{newErrorEvent()},
		},
		{
			name: "watchlist missing initial event",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2 /*addPod2RV3,*/, newInitialEventsEndBookmark("3"), bluePod1RV4},
			expectError: true,
		},
		{
			name: "watchlist modified event in initial events",
			requests: []WatchRequest{
				watchList("4", storage.Everything),
			},
			events:      []watch.Event{bluePod1RV4, addPod2RV3, newInitialEventsEndBookmark("4")},
			expectError: true,
		},
		{
			name: "watchlist missing bookmark",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3},
			expectError: true,
		},
		{
			name: "watchlist bookmark older than requested resource version",
			requests: []WatchRequest{
				watchList("4", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4},
			expectError: true,
		},
		{
			name: "watchlist missing event after bookmark",
			requests: []WatchRequest{
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3") /*bluePod1RV4,*/, deletePod2RV5},
			expectError: true,
		},
		{
			name: "watchlist event at bookmark resource version after bookmark",
			requests: []WatchRequest{
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), addPod2RV3, bluePod1RV4},
			expectError: true,
		},
		{
			name: "watchlist with no events",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("0", storage.Everything),
				watchList("3", isBlue),
			},
			events: []watch.Event{},
		},
		{
			name: "watchlist with periodic bookmark after initial events end bookmark",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("3", storage.Everything),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4, newBookmark("4")},
		},
		{
			name: "watchlist ending with error event after initial events",
			requests: []WatchRequest{
				watchList("3", storage.Everything),
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4, newErrorEvent()},
		},
		{
			name: "watchlist ending with error event before initial events end bookmark",
			requests: []WatchRequest{
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, newErrorEvent()},
			expectError: true,
		},
		{
			name: "watchlist bookmark without initial events end annotation",
			requests: []WatchRequest{
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newBookmark("3"), bluePod1RV4},
			expectError: true,
		},
		{
			name: "watchlist with repeated initial events end bookmark",
			requests: []WatchRequest{
				watchList("", storage.Everything),
				watchList("3", storage.Everything),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4, newInitialEventsEndBookmark("4")},
			expectError: true,
		},
		{
			name: "watch on a single key",
			requests: []WatchRequest{
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything}},
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact, Predicate: storage.Everything}},
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: storage.Everything}},
			},
			events: []watch.Event{addPod1RV2, bluePod1RV4, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "watch on a single key with event for another key",
			requests: []WatchRequest{
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything}},
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact, Predicate: storage.Everything}},
				{Key: pod1Key, Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: storage.Everything}},
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4},
			expectError: true,
		},
		{
			name: "watch on a namespace",
			requests: []WatchRequest{
				{Key: "/pods/ns1", Options: storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns1", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact, Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns1", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns1/", Options: storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns1/", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact, Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns1/", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: storage.Everything, Recursive: true}},
			},
			events: []watch.Event{addPod1RV2, addPod2RV3, bluePod1RV4, deletePod2RV5, redPod1RV6, pod1RV7, deletePod1RV8},
		},
		{
			name: "watch on another namespace with event from outside of it",
			requests: []WatchRequest{
				{Key: "/pods/ns", Options: storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact, Predicate: storage.Everything, Recursive: true}},
				{Key: "/pods/ns", Options: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: storage.Everything, Recursive: true}},
			},
			events:      []watch.Event{addPod1RV2},
			expectError: true,
		},
		{
			name: "watchlist on a single key",
			requests: []WatchRequest{
				onKey(pod1Key, watchList("3", storage.Everything)),
			},
			events: []watch.Event{addPod1RV2, newInitialEventsEndBookmark("3"), bluePod1RV4},
		},
		{
			name: "watchlist on a single key with initial event for another key",
			requests: []WatchRequest{
				onKey(pod1Key, watchList("3", storage.Everything)),
			},
			events:      []watch.Event{addPod1RV2, addPod2RV3, newInitialEventsEndBookmark("3"), bluePod1RV4},
			expectError: true,
		},
	}
	versioner := storage.APIObjectVersioner{}
	initialState := NewEmptyModel("", func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, versioner)
	replay, err := NewReplay(initialState, operations)
	require.NoError(t, err)
	validator := NewWatchValidator(versioner, replay, getKey)
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

var (
	isBlue = storage.SelectionPredicate{
		Label:    labels.SelectorFromSet(labels.Set{"color": "blue"}),
		Field:    fields.Everything(),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isRed = storage.SelectionPredicate{
		Label:    labels.SelectorFromSet(labels.Set{"color": "red"}),
		Field:    fields.Everything(),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isPod1 = storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.OneTermEqualSelector("metadata.name", "pod1"),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
	isPod2 = storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.OneTermEqualSelector("metadata.name", "pod2"),
		GetAttrs: storage.DefaultNamespaceScopedAttr,
	}
)

func watchEverything(rv string, match metav1.ResourceVersionMatch) WatchRequest {
	opts := storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: storage.Everything, Recursive: true, SendInitialEvents: new(false)}
	return WatchRequest{Key: "/pods/", Options: opts}
}

func watchBlue(rv string, match metav1.ResourceVersionMatch) WatchRequest {
	opts := storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: isBlue, Recursive: true, SendInitialEvents: new(false)}
	return WatchRequest{Key: "/pods/", Options: opts}
}

func watchRed(rv string, match metav1.ResourceVersionMatch) WatchRequest {
	opts := storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: isRed, Recursive: true, SendInitialEvents: new(false)}
	return WatchRequest{Key: "/pods/", Options: opts}
}

func watchPod1(rv string, match metav1.ResourceVersionMatch) WatchRequest {
	opts := storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: isPod1, Recursive: true, SendInitialEvents: new(false)}
	return WatchRequest{Key: "/pods/", Options: opts}
}

func watchPod2(rv string, match metav1.ResourceVersionMatch) WatchRequest {
	opts := storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: isPod2, Recursive: true, SendInitialEvents: new(false)}
	return WatchRequest{Key: "/pods/", Options: opts}
}

func watchList(rv string, pred storage.SelectionPredicate) WatchRequest {
	pred.AllowWatchBookmarks = true
	return WatchRequest{Key: "/pods/", Options: storage.ListOptions{
		ResourceVersion:      rv,
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		Predicate:            pred,
		Recursive:            true,
		SendInitialEvents:    new(true),
	}}
}

func onKey(key string, request WatchRequest) WatchRequest {
	request.Key = key
	request.Options.Recursive = false
	return request
}

func newBookmark(rv string) watch.Event {
	return watch.Event{Type: watch.Bookmark, Object: newTestPod("", "", "", rv)}
}

func newInitialEventsEndBookmark(rv string) watch.Event {
	pod := newTestPod("", "", "", rv)
	pod.Annotations = map[string]string{metav1.InitialEventsAnnotationKey: "true"}
	return watch.Event{Type: watch.Bookmark, Object: pod}
}

func newErrorEvent() watch.Event {
	return watch.Event{Type: watch.Error, Object: &metav1.Status{Status: metav1.StatusFailure, Message: "watch error"}}
}
