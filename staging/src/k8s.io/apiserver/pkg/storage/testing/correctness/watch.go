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
	"math"
	"reflect"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// WatchValidator checks watch streams against the event history the model
// derived from the recorded operations.
type WatchValidator struct {
	versioner    storage.Versioner
	keyFunc      func(runtime.Object) (string, error)
	expectEvents []watch.Event
}

// NewWatchValidator returns a validator for the given event history. keyFunc
// must be the same one the operations that produced expectEvents were keyed by.
func NewWatchValidator(versioner storage.Versioner, keyFunc func(runtime.Object) (string, error), expectEvents []watch.Event) WatchValidator {
	return WatchValidator{versioner: versioner, keyFunc: keyFunc, expectEvents: expectEvents}
}

func (v WatchValidator) ValidateWatch(request WatchRequest, response WatchResponse) error {
	if err := v.validateReliable(request, response); err != nil {
		return fmt.Errorf("watch %+v: Broke reliable %w", request, err)
	}
	if err := v.validateBookmarks(response.Events); err != nil {
		return fmt.Errorf("watch %+v: Broke bookmarks %w", request, err)
	}
	return nil
}

func (v WatchValidator) validateReliable(request WatchRequest, response WatchResponse) error {
	if len(response.Events) == 0 {
		// Empty watch response is always correct, because watch is eventually consistent.
		return nil
	}
	rangeRV, err := watchRevisionRange(v.versioner, request, response.Events)
	if err != nil {
		return err
	}
	expected, err := v.filterEvents(request, rangeRV)
	if err != nil {
		return err
	}
	expectedRefs, err := v.toEventReference(expected)
	if err != nil {
		return err
	}
	gotEvents := filterOutBookmarks(response.Events)
	gotRefs, err := v.toEventReference(gotEvents)
	if err != nil {
		return err
	}
	if diff := cmp.Diff(expectedRefs, gotRefs); diff != "" {
		return fmt.Errorf("events over revisions [%d, %d) differ (-want +got):\n%s", rangeRV.Min, rangeRV.Max, diff)
	}

	for i := range gotEvents {
		if !reflect.DeepEqual(expected[i].Object, gotEvents[i].Object) {
			return fmt.Errorf("event %d: object differs (-want +got):\n%s", i, cmp.Diff(expected[i].Object, gotEvents[i].Object))
		}
	}
	return nil
}

func (v WatchValidator) validateBookmarks(events []watch.Event) error {
	lastBookmarkRV := uint64(0)
	for _, ev := range events {
		rv, err := objectRV(ev.Object, v.versioner)
		if err != nil {
			return err
		}
		if ev.Type == watch.Bookmark {
			if rv < lastBookmarkRV {
				return fmt.Errorf("bookmark revision %d is not greater than last bookmark revision %d", rv, lastBookmarkRV)
			}
			lastBookmarkRV = rv
			continue
		}
		if rv <= lastBookmarkRV {
			return fmt.Errorf("event revision %d is not greater than last bookmark revision %d", rv, lastBookmarkRV)
		}
	}
	return nil
}

type eventReference struct {
	Key             string
	Type            watch.EventType
	ResourceVersion uint64
}

func (v WatchValidator) toEventReference(events []watch.Event) ([]eventReference, error) {
	refs := make([]eventReference, 0, len(events))
	for _, event := range events {
		key, err := v.keyFunc(event.Object)
		if err != nil {
			return nil, err
		}
		rv, err := objectRV(event.Object, v.versioner)
		if err != nil {
			return nil, err
		}
		refs = append(refs, eventReference{
			Key:             key,
			Type:            event.Type,
			ResourceVersion: rv,
		})
	}
	return refs, nil
}

func (v WatchValidator) filterEvents(request WatchRequest, rvRange *ResourceVersionRange) ([]watch.Event, error) {
	filtered := make([]watch.Event, 0, len(v.expectEvents))
	for _, event := range v.expectEvents {
		rv, err := objectRV(event.Object, v.versioner)
		if err != nil {
			return nil, err
		}
		if rv >= rvRange.Min && rv < rvRange.Max {
			filtered = append(filtered, event)
		}
	}
	return filtered, nil
}

func watchRevisionRange(versioner storage.Versioner, request WatchRequest, events []watch.Event) (*ResourceVersionRange, error) {
	minRV := uint64(math.MaxUint64)
	maxRV := uint64(0)
	for _, ev := range events {
		rv, err := objectRV(ev.Object, versioner)
		if err != nil {
			return nil, err
		}
		minRV = min(minRV, rv)
		if ev.Type == watch.Bookmark {
			// A bookmark only promises everything up to its revision was sent.
			maxRV = max(maxRV, rv)
		} else {
			maxRV = max(maxRV, rv+1)
		}
	}
	if request.ResourceVersion != "0" && request.ResourceVersion != "" {
		requestedRV, err := versioner.ParseResourceVersion(request.ResourceVersion)
		if err != nil {
			return nil, err
		}
		minRV = requestedRV + 1
	}
	return &ResourceVersionRange{Min: minRV, Max: maxRV}, nil
}

// ResourceVersionRange is a range of resource versions [Min, Max)
type ResourceVersionRange struct {
	Min, Max uint64
}

func objectRV(obj runtime.Object, versioner storage.Versioner) (uint64, error) {
	acc, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	rv, err := versioner.ParseResourceVersion(acc.GetResourceVersion())
	if err != nil {
		return 0, err
	}
	return rv, nil
}

func filterOutBookmarks(events []watch.Event) []watch.Event {
	filtered := make([]watch.Event, 0, len(events))
	for _, event := range events {
		if event.Type == watch.Bookmark {
			continue
		}
		filtered = append(filtered, event)
	}
	return filtered
}
