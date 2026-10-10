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
	"strconv"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// Replay stores historical model states keyed by ResourceVersion and the
// sequence of changes produced by replaying history.
type Replay struct {
	newFunc   func() runtime.Object
	versioner storage.Versioner

	states  map[uint64]*Model
	changes []Change
}

// NewReplay creates a Replay initialized with initialState and replays history on it.
func NewReplay(initialState *Model, history []Operation) (*Replay, error) {
	state := initialState.Clone()
	states := map[uint64]*Model{
		state.ResourceVersion: state,
	}
	var changes []Change
	for i, op := range history {
		ok, next, change := state.Step(op.Request, op.Response)
		if !ok {
			return nil, fmt.Errorf("operation %d failed model step: req=%+v resp=%+v", i, op.Request, op.Response)
		}
		state = next
		states[state.ResourceVersion] = state
		if change != nil {
			changes = append(changes, *change)
		}
	}
	return &Replay{
		newFunc:   initialState.NewFunc,
		versioner: initialState.Versioner,
		states:    states,
		changes:   changes,
	}, nil
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
	reqRV, err := r.versioner.ParseResourceVersion(opts.ResourceVersion)
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
	consistency, _, _, err := ListReadConsistency("", r.versioner, req.List.Options)
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
	respRV, err := r.versioner.ParseResourceVersion(accessor.GetResourceVersion())
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

func (r *Replay) Watch(request WatchRequest) ([]watch.Event, error) {
	var events []watch.Event
	rv := r.ResourceVersion()
	requestedRV, err := r.versioner.ParseResourceVersion(request.Options.ResourceVersion)
	if err != nil {
		return nil, err
	}
	if request.Options.SendInitialEvents != nil && *request.Options.SendInitialEvents {
		initialEvents, err := r.InitialEvents(request, rv)
		if err != nil {
			return nil, err
		}
		events = append(events, initialEvents...)
		bookmark := r.newFunc()
		accessor, err := meta.Accessor(bookmark)
		if err != nil {
			return nil, err
		}
		accessor.SetResourceVersion(strconv.FormatUint(rv, 10))
		accessor.SetAnnotations(map[string]string{metav1.InitialEventsAnnotationKey: "true"})
		events = append(events, watch.Event{Type: watch.Bookmark, Object: bookmark})
		requestedRV = rv
	}
	streamEvents, err := r.EventsForRVs(request, &ResourceVersionRange{Min: requestedRV + 1, Max: rv + 1})
	if err != nil {
		return nil, err
	}
	events = append(events, streamEvents...)
	return events, nil
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
		watchEvent, err := change.toWatchEvent(r.versioner, request.Options.Predicate)
		if err != nil {
			return nil, err
		}
		if watchEvent != nil {
			filtered = append(filtered, *watchEvent)
		}
	}
	return filtered, nil
}

func (r *Replay) ResourceVersion() uint64 {
	return r.changes[len(r.changes)-1].ResourceVersion
}

func (r *Replay) LastWatchRV(request WatchRequest) (uint64, error) {
	for i := len(r.changes) - 1; i >= 0; i-- {
		change := r.changes[i]
		if !keyInScope(request.Key, request.Options.Recursive, change.Key) {
			continue
		}
		watchEvent, err := change.toWatchEvent(r.versioner, request.Options.Predicate)
		if err != nil {
			return 0, err
		}
		if watchEvent != nil {
			return change.ResourceVersion, nil
		}
	}
	return 0, nil
}
