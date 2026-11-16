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
	"reflect"

	"github.com/google/go-cmp/cmp"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// Replay is a single-threaded history of monotonically advancing Model states
// and changes, supporting reads and watch replay at arbitrary resource versions.
type Replay struct {
	state   *Model
	states  map[uint64]*Model
	changes []Change
}

// NewReplay creates a Replay initialized with initialState and replays history on it.
func NewReplay(initialState *Model, history []Operation) (*Replay, error) {
	state := initialState.Clone()
	r := &Replay{
		state:  state,
		states: map[uint64]*Model{state.ResourceVersion: state},
	}
	for i, op := range history {
		ok, next, change := r.state.Step(op.Request, op.Response)
		if !ok {
			return nil, fmt.Errorf("operation %d failed model step: req=%+v resp=%+v", i, op.Request, op.Response)
		}
		r.recordState(next, change)
	}
	return r, nil
}

func (r *Replay) recordState(next *Model, change *Change) {
	if next.ResourceVersion < r.state.ResourceVersion {
		panic(fmt.Sprintf("resource version went backwards from %d to %d", r.state.ResourceVersion, next.ResourceVersion))
	}
	if next.CompactResourceVersion < r.state.CompactResourceVersion {
		panic(fmt.Sprintf("compact resource version went backwards from %d to %d", r.state.CompactResourceVersion, next.CompactResourceVersion))
	}
	if change != nil {
		if change.ResourceVersion != next.ResourceVersion || change.ResourceVersion <= r.state.ResourceVersion {
			panic(fmt.Sprintf("invalid change resource version %d (prev=%d, next=%d)", change.ResourceVersion, r.state.ResourceVersion, next.ResourceVersion))
		}
		r.changes = append(r.changes, *change)
	}
	r.state = next
	r.states[next.ResourceVersion] = next
}

func (r *Replay) execute(req Request) (Response, *Change) {
	resp, next, change := r.state.execute(req)
	if resp.Err == nil {
		r.recordState(next, change)
	}
	if resp.Err == nil && req.Op == OpList {
		if consistency, rv, _, err := ListReadConsistency("", r.state.Versioner, req.List.Options); err == nil && consistency == ConsistencyExact {
			if state, ok := r.states[rv]; ok {
				resp = state.list(req.Key, req.List.Options)
			}
		}
	}
	return resp, change
}

func (r *Replay) increaseRV() uint64 {
	next := r.state.Clone()
	next.ResourceVersion++
	r.recordState(next, nil)
	return r.state.ResourceVersion
}

// Validate checks that the response of a non-consistent read matches the
// response produced by executing req on the model state at the response's RV.
func (r *Replay) Validate(req Request, resp Response) error {
	switch req.Op {
	case OpGet:
		return r.validateGet(req, resp)
	case OpList:
		return r.validateList(req, resp)
	default:
		return nil
	}
}

func (r *Replay) validateGet(req Request, resp Response) error {
	opts := req.Get.Options
	consistency := GetReadConsistency(opts)
	if consistency == ConsistencyConsistent {
		return nil
	}
	reqRV, err := r.state.Versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil || storage.IsTooLargeResourceVersion(resp.Err) {
		return nil
	}
	for rv, state := range r.states {
		if rv < reqRV {
			continue
		}
		// Cacher returns NotFound errors without the underlying etcd prefix.
		stateWithoutPrefix := *state
		stateWithoutPrefix.Prefix = ""
		if reflect.DeepEqual(state.get(req.Key, opts), resp) || reflect.DeepEqual(stateWithoutPrefix.get(req.Key, opts), resp) {
			return nil
		}
	}
	return fmt.Errorf("get(%s, RV=%s) response %+v does not match any state at RV >= %d", req.Key, opts.ResourceVersion, resp, reqRV)
}

func (r *Replay) validateList(req Request, resp Response) error {
	if resp.Err != nil {
		return nil
	}
	consistency, _, _, err := ListReadConsistency("", r.state.Versioner, req.List.Options)
	if err != nil {
		return err
	}
	if consistency == ConsistencyConsistent {
		return nil
	}
	if resp.Object == nil {
		return fmt.Errorf("expected non-nil list object")
	}
	accessor, err := meta.ListAccessor(resp.Object)
	if err != nil {
		return err
	}
	respRV, err := r.state.Versioner.ParseResourceVersion(accessor.GetResourceVersion())
	if err != nil {
		return err
	}
	state, ok := r.states[respRV]
	if !ok {
		return fmt.Errorf("resource version %d not found in history", respRV)
	}
	expected := state.list(req.Key, req.List.Options)
	if !state.equalListResponse(req.Key, req.List.Options, expected, resp) {
		return fmt.Errorf("list response at RV %d differs (-want +got):\n%s", respRV, cmp.Diff(expected, resp))
	}
	return nil
}

// Watch opens request against the historical state at openRV and replays all
// subsequent matching events up to r.ResourceVersion().
func (r *Replay) Watch(request WatchRequest, openRV uint64) ([]watch.Event, error) {
	events, nextRV, closed, err := r.watch(request, openRV)
	if err != nil || closed {
		return events, err
	}
	rest, err := r.EventsForRVs(request, &ResourceVersionRange{Min: nextRV, Max: r.ResourceVersion() + 1})
	events = append(events, rest...)
	if err != nil {
		events = append(events, errorWatchEvent(err))
	}
	return events, nil
}

// watch evaluates opening request against the historical state at openRV,
// returning initial or catchup events up to openRV, the next resource version
// to stream after openRV, and whether the watch terminates after those events.
func (r *Replay) watch(request WatchRequest, openRV uint64) (events []watch.Event, nextRV uint64, closed bool, err error) {
	openState, ok := r.states[openRV]
	if !ok {
		return nil, 0, false, fmt.Errorf("resource version %d not found in history", openRV)
	}
	if err := checkWatchOptions(openState.Versioner, request); err != nil {
		return nil, 0, false, err
	}
	opts := request.Options
	requestedRV, err := openState.Versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return nil, 0, false, err
	}
	sendInitial := opts.SendInitialEvents != nil && *opts.SendInitialEvents
	legacyInitial := opts.SendInitialEvents == nil && requestedRV == 0
	if sendInitial || legacyInitial {
		if sendInitial && requestedRV > openRV {
			return []watch.Event{errorWatchEvent(storage.NewTooLargeResourceVersionError(requestedRV, openRV, 1))}, 0, true, nil
		}
		initial, err := openState.initialEvents(request)
		if err != nil {
			return []watch.Event{errorWatchEvent(err)}, 0, true, nil
		}
		events = append(events, initial...)
		if sendInitial && opts.Predicate.AllowWatchBookmarks {
			bookmark, err := newBookmarkEvent(openState.NewFunc, openState.Versioner, openRV, true)
			if err != nil {
				return []watch.Event{errorWatchEvent(err)}, 0, true, nil
			}
			events = append(events, bookmark)
		}
		return events, openRV + 1, false, nil
	}
	if requestedRV == 0 {
		return nil, openRV + 1, false, nil
	}
	if requestedRV < openState.CompactResourceVersion {
		return []watch.Event{errorWatchEvent(apierrors.NewResourceExpired("The resourceVersion for the provided watch is too old."))}, 0, true, nil
	}
	catchup, err := r.EventsForRVs(request, &ResourceVersionRange{Min: requestedRV + 1, Max: openRV + 1})
	events = append(events, catchup...)
	if err != nil {
		events = append(events, errorWatchEvent(err))
		return events, 0, true, nil
	}
	if opts.ProgressNotify && openRV > requestedRV {
		bookmark, err := newBookmarkEvent(openState.NewFunc, openState.Versioner, openRV, false)
		if err != nil {
			events = append(events, errorWatchEvent(err))
			return events, 0, true, nil
		}
		events = append(events, bookmark)
	}
	return events, max(requestedRV, openRV) + 1, false, nil
}

func (r *Replay) InitialEvents(request WatchRequest, rv uint64) ([]watch.Event, error) {
	state, ok := r.states[rv]
	if !ok {
		return nil, fmt.Errorf("resource version %d not found in history", rv)
	}
	return state.initialEvents(request)
}

func (r *Replay) EventsForRVs(request WatchRequest, rvRange *ResourceVersionRange) ([]watch.Event, error) {
	filtered := make([]watch.Event, 0, len(r.changes))
	for _, change := range r.changes {
		if change.ResourceVersion < rvRange.Min || change.ResourceVersion >= rvRange.Max {
			continue
		}
		if !keyInScope(request.Key, request.Options.Recursive, change.Key) {
			continue
		}
		watchEvent, err := change.toWatchEvent(r.state.NewFunc, r.state.Versioner, r.state.ReverseKeyFunc, request.Options)
		if err != nil {
			return filtered, err
		}
		if watchEvent != nil {
			filtered = append(filtered, *watchEvent)
		}
	}
	return filtered, nil
}

func (r *Replay) ResourceVersion() uint64 {
	return r.state.ResourceVersion
}
