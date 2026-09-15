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
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

// ValidateWatchGuarantees verifies the formal watch guarantees inspired by etcd robustness testing:
// 1. Ordered: Event resource versions increase monotonically without backwards jumps.
// 2. Unique: No duplicate events for the same key at the same resource version.
// 3. Reliable: The watcher observes all committed mutating events matching its prefix and predicate in exact order.
// 4. Resumable: A watch starting from revision RV receives the very next committed event after RV.
// 5. WatchList: Streaming events following the initial-events-end bookmark match committed events from bookmark RV.
func ValidateWatchGuarantees(t *testing.T, versioner storage.Versioner, history *WatchHistory, req WatchRequest, resp WatchResponse) {
	t.Helper()
	if versioner == nil {
		versioner = storage.APIObjectVersioner{}
	}

	events := resp.Events
	if len(events) == 0 {
		return
	}

	watchName := req.Name
	if watchName == "" {
		watchName = req.Key
	}

	var added, modified, deleted, bookmarks, errors int
	for _, ev := range events {
		switch ev.Type {
		case watch.Added:
			added++
		case watch.Modified:
			modified++
		case watch.Deleted:
			deleted++
		case watch.Bookmark:
			bookmarks++
		case watch.Error:
			errors++
		}
	}

	t.Logf("Watch[%s]: %d total events read (startRV=%s, sendInitialEvents=%t) -> [Added: %d, Modified: %d, Deleted: %d, Bookmarks: %d, Errors: %d]",
		watchName, len(events), req.ResourceVersion, req.SendInitialEvents, added, modified, deleted, bookmarks, errors)

	if req.SendInitialEvents {
		validateWatchListStream(t, versioner, history, req, resp, watchName)
		return
	}
	validateStandardWatchStream(t, versioner, history, req, resp, watchName)
}

func validateStandardWatchStream(t *testing.T, versioner storage.Versioner, history *WatchHistory, req WatchRequest, resp WatchResponse, watchName string) {
	t.Helper()

	events := resp.Events
	var mutatingEvents []watch.Event
	for _, ev := range events {
		if ev.Type == watch.Added || ev.Type == watch.Modified || ev.Type == watch.Deleted {
			mutatingEvents = append(mutatingEvents, ev)
		}
	}

	// 1. Ordered guarantee across all events
	var lastRV uint64
	for i, ev := range events {
		if ev.Type == watch.Error {
			continue
		}
		acc, err := meta.Accessor(ev.Object)
		require.NoError(t, err, "watch[%s] event[%d] missing ObjectMeta", watchName, i)
		evRV, err := versioner.ParseResourceVersion(acc.GetResourceVersion())
		require.NoError(t, err, "watch[%s] event[%d] invalid RV: %s", watchName, i, acc.GetResourceVersion())

		require.True(t, evRV >= lastRV, "watch[%s] broke ORDERED guarantee: event[%d] RV %d < previous RV %d", watchName, i, evRV, lastRV)
		lastRV = evRV
	}

	// 2. Unique guarantee across mutating events
	type eventKey struct {
		rv  uint64
		key string
	}
	seenEvents := make(map[eventKey]struct{})
	for _, ev := range mutatingEvents {
		acc, _ := meta.Accessor(ev.Object)
		evRV, _ := versioner.ParseResourceVersion(acc.GetResourceVersion())
		key := fmt.Sprintf("/pods/%s/%s", acc.GetNamespace(), acc.GetName())
		k := eventKey{rv: evRV, key: key}

		_, duplicate := seenEvents[k]
		require.False(t, duplicate, "watch[%s] broke UNIQUE guarantee: duplicate event for key %s at RV %d", watchName, key, evRV)
		seenEvents[k] = struct{}{}
	}

	// 3. Reliable guarantee against replayed linearized history
	startRV, _ := versioner.ParseResourceVersion(req.ResourceVersion)
	expectedMatchingEvents := history.ExpectedEvents(req.Key, startRV, req.Predicate)

	startIdx := 0
	if startRV == 0 && len(mutatingEvents) > 0 {
		firstActMeta, _ := meta.Accessor(mutatingEvents[0].Object)
		firstActRV, _ := versioner.ParseResourceVersion(firstActMeta.GetResourceVersion())
		found := false
		for idx, exp := range expectedMatchingEvents {
			expMeta, _ := meta.Accessor(exp.Object)
			expRV, _ := versioner.ParseResourceVersion(expMeta.GetResourceVersion())
			if expRV == firstActRV && exp.Type == mutatingEvents[0].Type && expMeta.GetName() == firstActMeta.GetName() && expMeta.GetNamespace() == firstActMeta.GetNamespace() {
				startIdx = idx
				found = true
				break
			}
		}
		require.True(t, found, "watch[%s] first event (RV=%d, Type=%v, Name=%s/%s) not found in expected history",
			watchName, firstActRV, mutatingEvents[0].Type, firstActMeta.GetNamespace(), firstActMeta.GetName())
	}

	require.True(t, startIdx+len(mutatingEvents) <= len(expectedMatchingEvents),
		"watch[%s] received more events (%d from idx %d) than expected matching events (%d)",
		watchName, len(mutatingEvents), startIdx, len(expectedMatchingEvents))

	for i := range mutatingEvents {
		exp := expectedMatchingEvents[startIdx+i]
		act := mutatingEvents[i]

		require.Equal(t, exp.Type, act.Type, "watch[%s] event[%d] type mismatch", watchName, i)

		expMeta, _ := meta.Accessor(exp.Object)
		actMeta, _ := meta.Accessor(act.Object)

		require.Equal(t, expMeta.GetNamespace(), actMeta.GetNamespace(), "watch[%s] event[%d] namespace mismatch", watchName, i)
		require.Equal(t, expMeta.GetName(), actMeta.GetName(), "watch[%s] event[%d] name mismatch", watchName, i)
		expRV, _ := versioner.ParseResourceVersion(expMeta.GetResourceVersion())
		actRV, _ := versioner.ParseResourceVersion(actMeta.GetResourceVersion())
		require.Equal(t, expRV, actRV, "watch[%s] event[%d] RV mismatch: expType=%v actType=%v expName=%s actName=%s", watchName, i, exp.Type, act.Type, expMeta.GetName(), actMeta.GetName())
	}

	if len(mutatingEvents) < len(expectedMatchingEvents) {
		t.Logf("watch[%s] terminated cleanly after %d / %d events at revision %d",
			watchName, len(mutatingEvents), len(expectedMatchingEvents), lastRV)
	}

	// 4. Resumable guarantee
	if startRV > 0 && len(expectedMatchingEvents) > 0 && len(mutatingEvents) > 0 {
		firstActMeta, _ := meta.Accessor(mutatingEvents[0].Object)
		firstActRV, _ := versioner.ParseResourceVersion(firstActMeta.GetResourceVersion())
		firstExpMeta, _ := meta.Accessor(expectedMatchingEvents[0].Object)
		firstExpRV, _ := versioner.ParseResourceVersion(firstExpMeta.GetResourceVersion())

		require.Equal(t, firstExpRV, firstActRV, "watch[%s] broke RESUMABLE guarantee: first event RV %d != expected first RV %d after startRV %d",
			watchName, firstActRV, firstExpRV, startRV)
	}
}

func validateWatchListStream(t *testing.T, versioner storage.Versioner, history *WatchHistory, req WatchRequest, resp WatchResponse, watchName string) {
	t.Helper()

	events := resp.Events
	var bookmarkIdx = -1
	var bookmarkRV uint64

	for i, ev := range events {
		if ev.Type == watch.Bookmark {
			isInitialEnd, err := storage.HasInitialEventsEndBookmarkAnnotation(ev.Object)
			if err == nil && isInitialEnd {
				bookmarkIdx = i
				acc, aErr := meta.Accessor(ev.Object)
				require.NoError(t, aErr)
				rv, pErr := versioner.ParseResourceVersion(acc.GetResourceVersion())
				require.NoError(t, pErr)
				bookmarkRV = rv
				break
			}
		}
	}

	if bookmarkIdx < 0 {
		t.Logf("WatchList[%s]: initial-events-end bookmark not found (validating as standard stream)", watchName)
		validateStandardWatchStream(t, versioner, history, req, resp, watchName)
		return
	}
	t.Logf("WatchList[%s]: initial-events-end bookmark received at index %d (RV=%d), validating streaming events", watchName, bookmarkIdx, bookmarkRV)

	// Ignore events prior to the initial-events-end bookmark and validate streaming events thereafter.
	streamingEvents := events[bookmarkIdx+1:]
	reqStreaming := req
	reqStreaming.ResourceVersion = fmt.Sprintf("%d", bookmarkRV)
	reqStreaming.SendInitialEvents = false
	respStreaming := WatchResponse{Events: streamingEvents}
	validateStandardWatchStream(t, versioner, history, reqStreaming, respStreaming, watchName+"-streaming")
}

// WatchHistory tracks committed mutations and computes expected watch events.
type WatchHistory struct {
	CommittedOps []Operation
}

// ReplayState is an alias for WatchHistory.
type ReplayState = WatchHistory

// NewWatchHistory constructs a WatchHistory from a list of executed operations.
func NewWatchHistory(ops []Operation, versioner storage.Versioner) *WatchHistory {
	if versioner == nil {
		versioner = storage.APIObjectVersioner{}
	}
	history := &WatchHistory{}
	for _, op := range ops {
		if op.Response.Err != nil || op.Response.Object == nil {
			continue
		}
		switch op.Request.Op {
		case OpCreate, OpUpdate, OpDelete:
			history.CommittedOps = append(history.CommittedOps, op)
		}
	}
	sort.Slice(history.CommittedOps, func(i, j int) bool {
		accI, errI := meta.Accessor(history.CommittedOps[i].Response.Object)
		accJ, errJ := meta.Accessor(history.CommittedOps[j].Response.Object)
		if errI != nil || errJ != nil {
			return false
		}
		rvI, _ := versioner.ParseResourceVersion(accI.GetResourceVersion())
		rvJ, _ := versioner.ParseResourceVersion(accJ.GetResourceVersion())
		return rvI < rvJ
	})
	return history
}

// ExpectedEvents constructs the exact expected watch events from the operation history for a given watch configuration.
func (h *WatchHistory) ExpectedEvents(prefix string, startRV uint64, pred storage.SelectionPredicate) []watch.Event {
	var expected []watch.Event
	isRecursive := strings.HasSuffix(prefix, "/")
	versioner := storage.APIObjectVersioner{}

	type keyState struct {
		matched bool
		obj     runtime.Object
	}
	trackedState := make(map[string]keyState)

	for _, op := range h.CommittedOps {
		req := op.Request
		res := op.Response

		acc, err := meta.Accessor(res.Object)
		if err != nil {
			continue
		}
		rv, _ := versioner.ParseResourceVersion(acc.GetResourceVersion())

		if rv <= startRV {
			if req.Op == OpDelete {
				delete(trackedState, req.Key)
			} else if res.Object != nil {
				trackedState[req.Key] = keyState{
					matched: matchKey(req.Key, prefix, isRecursive) && matchesPredicate(res.Object, pred),
					obj:     res.Object.DeepCopyObject(),
				}
			}
			continue
		}

		if !matchKey(req.Key, prefix, isRecursive) {
			continue
		}

		prev, wasTracked := trackedState[req.Key]
		matchedBefore := wasTracked && prev.matched

		switch req.Op {
		case OpCreate:
			matchedNow := matchesPredicate(res.Object, pred)
			if matchedNow {
				expected = append(expected, watch.Event{
					Type:   watch.Added,
					Object: res.Object.DeepCopyObject(),
				})
			}
			trackedState[req.Key] = keyState{matched: matchedNow, obj: res.Object.DeepCopyObject()}

		case OpUpdate:
			matchedNow := matchesPredicate(res.Object, pred)
			if !matchedBefore && matchedNow {
				expected = append(expected, watch.Event{
					Type:   watch.Added,
					Object: res.Object.DeepCopyObject(),
				})
			} else if matchedBefore && matchedNow {
				expected = append(expected, watch.Event{
					Type:   watch.Modified,
					Object: res.Object.DeepCopyObject(),
				})
			} else if matchedBefore && !matchedNow {
				delObj := prev.obj.DeepCopyObject()
				_ = versioner.UpdateObject(delObj, rv)
				expected = append(expected, watch.Event{
					Type:   watch.Deleted,
					Object: delObj,
				})
			}
			trackedState[req.Key] = keyState{matched: matchedNow, obj: res.Object.DeepCopyObject()}

		case OpDelete:
			if matchedBefore {
				expected = append(expected, watch.Event{
					Type:   watch.Deleted,
					Object: res.Object.DeepCopyObject(),
				})
			}
			delete(trackedState, req.Key)
		}
	}
	return expected
}

// CreatePodPredicate creates a SelectionPredicate for pods with default matching attributes.
func CreatePodPredicate(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	if label == nil {
		label = labels.Everything()
	}
	if field == nil {
		field = fields.Everything()
	}
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: PodAttrFunc,
	}
}

// PodAttrFunc returns the labels and fields for an example.Pod or generic runtime.Object.
func PodAttrFunc(obj runtime.Object) (labels.Set, fields.Set, error) {
	if obj == nil {
		return nil, nil, nil
	}
	pod, ok := obj.(*example.Pod)
	if !ok {
		return storage.DefaultNamespaceScopedAttr(obj)
	}
	return labels.Set(pod.Labels), fields.Set{
		"metadata.name":      pod.Name,
		"metadata.namespace": pod.Namespace,
		"spec.nodeName":      pod.Spec.NodeName,
	}, nil
}

func normalizePredicate(pred storage.SelectionPredicate) storage.SelectionPredicate {
	if pred.Label == nil {
		pred.Label = labels.Everything()
	}
	if pred.Field == nil {
		pred.Field = fields.Everything()
	}
	if pred.GetAttrs == nil {
		pred.GetAttrs = PodAttrFunc
	}
	return pred
}

func matchesPredicate(obj runtime.Object, pred storage.SelectionPredicate) bool {
	if pred.Empty() {
		return true
	}
	if obj == nil {
		return false
	}
	if pred.GetAttrs != nil {
		l, f, err := pred.GetAttrs(obj)
		if err != nil {
			return false
		}
		if pred.Label != nil && !pred.Label.Matches(l) {
			return false
		}
		if pred.Field != nil && !pred.Field.Matches(f) {
			return false
		}
		return true
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false
	}
	if pred.Label != nil && !pred.Label.Matches(labels.Set(accessor.GetLabels())) {
		return false
	}
	if pred.Field != nil && !pred.Field.Empty() {
		pod, ok := obj.(*example.Pod)
		if ok {
			fieldsSet := fields.Set{
				"metadata.name":      pod.Name,
				"metadata.namespace": pod.Namespace,
				"spec.nodeName":      pod.Spec.NodeName,
			}
			if !pred.Field.Matches(fieldsSet) {
				return false
			}
		} else {
			fieldsSet := fields.Set{
				"metadata.name":      accessor.GetName(),
				"metadata.namespace": accessor.GetNamespace(),
			}
			if !pred.Field.Matches(fieldsSet) {
				return false
			}
		}
	}
	return true
}

func matchKey(key, prefix string, recursive bool) bool {
	if !recursive {
		return key == prefix
	}
	if !strings.HasSuffix(prefix, "/") {
		prefix = prefix + "/"
	}
	if key == strings.TrimSuffix(prefix, "/") {
		return true
	}
	return strings.HasPrefix(key, prefix)
}
