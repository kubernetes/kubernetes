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
	"math"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

func ValidateWatch(t *testing.T, versioner storage.Versioner, expectedEvents []watch.Event, watchRequest WatchRequest, watchResp []watch.Event) {
	t.Helper()
	if versioner == nil {
		versioner = storage.APIObjectVersioner{}
	}
	if len(watchResp) == 0 {
		return
	}
	minRV, maxRV := minMaxRV(t, watchResp, versioner)
	if watchRequest.ResourceVersion != "0" && watchRequest.ResourceVersion != "" {
		requestedRV, err := versioner.ParseResourceVersion(watchRequest.ResourceVersion)
		require.NoError(t, err)
		minRV = requestedRV + 1
	}

	expectedMatchingEvents := eventsForRange(t, expectedEvents, minRV, maxRV, versioner)
	require.Len(t, watchResp, len(expectedMatchingEvents), "watch events don't match the operation history replayed over [%d, %d]", minRV, maxRV)
	for i := range watchResp {
		require.Equal(t, expectedMatchingEvents[i].Type, watchResp[i].Type, "event %d type differs", i)
		require.Equal(t, expectedMatchingEvents[i].Object, watchResp[i].Object, "event %d object differs", i)
	}
}

// UnwrapEvent returns the event with the object the store was asked to store.
// The cacher may serve a wrapper that memoizes the object's serialization,
// which is invisible over the wire but not to reflect.DeepEqual. Callers
// recording a watch stream should normalize events through this.
func UnwrapEvent(event watch.Event) watch.Event {
	if cacheable, ok := event.Object.(runtime.CacheableObject); ok {
		event.Object = cacheable.GetObject()
	}
	return event
}

func minMaxRV(t *testing.T, events []watch.Event, versioner storage.Versioner) (uint64, uint64) {
	t.Helper()
	minRV := uint64(math.MaxUint64)
	maxRV := uint64(0)
	for _, ev := range events {
		if ev.Type == watch.Error {
			continue
		}
		accesor, err := meta.Accessor(ev.Object)
		require.NoError(t, err)
		rv, err := versioner.ParseResourceVersion(accesor.GetResourceVersion())
		require.NoError(t, err)
		minRV = min(minRV, rv)
		maxRV = max(maxRV, rv)
	}
	return minRV, maxRV
}

// OperationsToWatch replays the operation history as the watch event stream a
// watcher observing the whole keyspace from the beginning is expected to see.
func OperationsToWatch(ops []Operation, versioner storage.Versioner) []watch.Event {
	type write struct {
		op Operation
		rv uint64
	}
	writes := make([]write, 0, len(ops))
	for _, op := range ops {
		switch op.Request.Op {
		case OpCreate, OpUpdate, OpDelete:
		default:
			continue
		}
		if op.Response.Err != nil {
			continue
		}
		writes = append(writes, write{op: op, rv: operationRV(op, versioner)})
	}
	sort.SliceStable(writes, func(i, j int) bool {
		if writes[i].rv != writes[j].rv {
			return writes[i].rv < writes[j].rv
		}
		// Ties are a no-op update sharing a revision with the write that
		// produced it, in whichever order the two happened to be recorded.
		// Put the real write first so the dedupe below keeps it.
		return writes[i].op.Request.Op != OpUpdate && writes[j].op.Request.Op == OpUpdate
	})

	type keyRevision struct {
		key string
		rv  uint64
	}
	seen := make(map[keyRevision]bool, len(writes))
	exists := make(map[string]bool)
	events := make([]watch.Event, 0, len(writes))
	for _, w := range writes {
		// An update that doesn't change the object isn't written to the etcd and doesn't generate an event.
		id := keyRevision{key: w.op.Request.Key, rv: w.rv}
		if seen[id] {
			continue
		}
		seen[id] = true

		var eventType watch.EventType
		switch w.op.Request.Op {
		case OpCreate:
			eventType = watch.Added
		case OpDelete:
			eventType = watch.Deleted
		case OpUpdate:
			// An update with IgnoreNotFound on a missing key creates the
			// object, which the store reports as an add.
			if exists[w.op.Request.Key] {
				eventType = watch.Modified
			} else {
				eventType = watch.Added
			}
		}
		exists[w.op.Request.Key] = w.op.Request.Op != OpDelete

		events = append(events, watch.Event{
			Type:   eventType,
			Object: w.op.Response.Object.DeepCopyObject(),
		})
	}
	return events
}

func operationRV(op Operation, versioner storage.Versioner) uint64 {
	acc, err := meta.Accessor(op.Response.Object)
	if err != nil {
		panic(err)
	}
	rv, err := versioner.ParseResourceVersion(acc.GetResourceVersion())
	if err != nil {
		panic(err)
	}
	return rv
}

func eventsForRange(t *testing.T, events []watch.Event, min, max uint64, versioner storage.Versioner) []watch.Event {
	filtered := make([]watch.Event, 0, len(events))
	for _, event := range events {
		if event.Type == watch.Error {
			filtered = append(filtered, event)
			continue
		}
		accesor, err := meta.Accessor(event.Object)
		require.NoError(t, err)
		rv, err := versioner.ParseResourceVersion(accesor.GetResourceVersion())
		if err != nil {
			panic(err)
		}
		if rv >= min && rv <= max {
			filtered = append(filtered, event)
		}
	}
	return filtered
}
