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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// Replay stores historical model states keyed by ResourceVersion and the
// sequence of changes produced by replaying history.
type Replay struct {
	versioner storage.Versioner
	states    map[uint64]*Model
	changes   []Change
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
		versioner: initialState.Versioner,
		states:    states,
		changes:   changes,
	}, nil
}

// Validate checks that the response of a non-consistent read matches the
// response produced by executing req on the model state at the response's RV.
func (r *Replay) Validate(req Request, resp Response) error {
	if req.Op != OpList || req.List.Options.ResourceVersion == "" || resp.Err != nil {
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
	if !reflect.DeepEqual(expected, resp) {
		return fmt.Errorf("list response at RV %d differs (-want +got):\n%s", respRV, cmp.Diff(expected, resp))
	}
	return nil
}

func (r *Replay) Events(request WatchRequest, rvRange *ResourceVersionRange) ([]watch.Event, error) {
	filtered := make([]watch.Event, 0, len(r.changes))
	for _, change := range r.changes {
		if change.ResourceVersion < rvRange.Min || change.ResourceVersion >= rvRange.Max {
			continue
		}
		watchEvent, err := change.toWatchEvent(r.versioner, request.Predicate)
		if err != nil {
			return nil, err
		}
		if watchEvent != nil {
			filtered = append(filtered, *watchEvent)
		}
	}
	return filtered, nil
}
