/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

func ExpectWatchEventsExact(events []watch.Event) *WatchExpectation {
	return &WatchExpectation{
		mode:   watchMatchExact,
		events: events,
	}
}

func ExpectWatchEventsPrefix(events []watch.Event, minRVObj, maxRVObj v1.Object) *WatchExpectation {
	return &WatchExpectation{
		mode:     watchMatchPrefixOpen,
		events:   events,
		minRVObj: minRVObj,
		maxRVObj: maxRVObj,
	}
}

func ExpectWatchEventsPrefixAndClose(events []watch.Event, minRVObj, maxRVObj v1.Object) *WatchExpectation {
	return &WatchExpectation{
		mode:     watchMatchPrefixClosed,
		events:   events,
		minRVObj: minRVObj,
		maxRVObj: maxRVObj,
	}
}

func ExpectWatchEventsPrefixAndCompacted(events []watch.Event, minRVObj, maxRVObj, compactObj v1.Object) *WatchExpectation {
	return &WatchExpectation{
		mode:       watchMatchPrefixCompacted,
		events:     events,
		minRVObj:   minRVObj,
		maxRVObj:   maxRVObj,
		compactObj: compactObj,
	}
}

type watchMatchMode int

const (
	watchMatchExact watchMatchMode = iota
	watchMatchPrefixOpen
	watchMatchPrefixClosed
	watchMatchPrefixCompacted
)

type WatchExpectation struct {
	mode       watchMatchMode
	events     []watch.Event
	minRVObj   v1.Object
	maxRVObj   v1.Object
	compactObj v1.Object
}

func (e *WatchExpectation) Check(t *testing.T, w watch.Interface) {
	t.Helper()
	if e.mode == watchMatchExact {
		testCheckResultsInStrictOrder(t, w, e.events)
		return
	}

	timeout := wait.ForeverTestTimeout
	if e.mode == watchMatchPrefixOpen {
		timeout = 100 * time.Millisecond
	}

	var (
		matched  int
		errEvent *watch.Event
		closed   bool
	)
loop:
	for i, expectedEvent := range e.events {
		select {
		case res, ok := <-w.ResultChan():
			if !ok {
				matched = i
				closed = true
				break loop
			}
			if co, ok := res.Object.(runtime.CacheableObject); ok {
				res.Object = co.GetObject()
			}
			if withRecordTime, ok := res.Object.(storage.WatchEventWithRecordTime); ok {
				res.Object = withRecordTime.Unwrap()
			}
			if res.Type == watch.Error {
				matched = i
				errEvent = &res
				testCheckStop(t, w)
				break loop
			}
			expectNoDiff(t, "incorrect event", expectedEvent, res)
			matched = i + 1
		case <-time.After(timeout):
			if e.mode != watchMatchPrefixOpen {
				t.Fatalf("time out after waiting %v on ResultChan", timeout)
			}
			break loop
		}
	}

	require.Positive(t, matched, "expected at least one event")
	lastObj, err := meta.Accessor(e.events[matched-1].Object)
	require.NoError(t, err)
	lastRV := mustParseRV(t, lastObj.GetResourceVersion())
	require.GreaterOrEqual(t, lastRV, mustParseRV(t, e.minRVObj.GetResourceVersion()))
	require.LessOrEqual(t, lastRV, mustParseRV(t, e.maxRVObj.GetResourceVersion()))

	switch e.mode {
	case watchMatchPrefixOpen:
		require.False(t, closed, "expected watch channel to remain open")
		require.Nil(t, errEvent)
	case watchMatchPrefixClosed:
		if !closed {
			testCheckStop(t, w)
		}
		require.Nil(t, errEvent)
	case watchMatchPrefixCompacted:
		require.NotNil(t, errEvent, "expected compacted error event before close")
		expectedErr := watch.Event{
			Type:   watch.Error,
			Object: &errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d (%s)", lastRV, e.compactObj.GetResourceVersion())).ErrStatus,
		}
		expectNoDiff(t, "incorrect event", expectedErr, *errEvent)
	}
}

func mustParseRV(t *testing.T, rv string) uint64 {
	t.Helper()
	parsed, err := strconv.ParseUint(rv, 10, 64)
	require.NoError(t, err)
	return parsed
}
