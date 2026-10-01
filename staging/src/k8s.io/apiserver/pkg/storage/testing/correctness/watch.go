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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// WatchValidator checks watch streams against the replayed model history.
type WatchValidator struct {
	versioner storage.Versioner
	keyFunc   func(runtime.Object) (string, error)
	replay    *Replay
}

// NewWatchValidator returns a validator for the given history of changes.
// keyFunc must be the same one the operations that produced history were keyed by.
func NewWatchValidator(versioner storage.Versioner, replay *Replay, keyFunc func(runtime.Object) (string, error)) WatchValidator {
	return WatchValidator{versioner: versioner, replay: replay, keyFunc: keyFunc}
}

func (v WatchValidator) ValidateWatch(request WatchRequest, response WatchResponse) error {
	if err := v.checkWatch(request); err != nil {
		if !reflect.DeepEqual(err, response.Err) {
			return fmt.Errorf("watch %+v: expected error %v, got %v", request, err, response.Err)
		}
		return nil
	}
	if response.Err != nil {
		return fmt.Errorf("watch %+v: unexpected error: %w", request, response.Err)
	}
	if err := validateMaybeLastEventError(response.Events); err != nil {
		return fmt.Errorf("watch %+v: Broke error %w", request, err)
	}
	if err := v.validateReliable(request, response); err != nil {
		return fmt.Errorf("watch %+v: Broke reliable %w", request, err)
	}
	if err := v.validateBookmarks(response.Events); err != nil {
		return fmt.Errorf("watch %+v: Broke bookmarks %w", request, err)
	}
	return nil
}

// checkWatch returns the error storage returns for an invalid watch and panics
// on watches the model can't reproduce.
func (v WatchValidator) checkWatch(request WatchRequest) error {
	opts := request.Options
	if err := checkKey(request.Key, opts.Recursive); err != nil {
		return err
	}
	if _, err := v.versioner.ParseResourceVersion(opts.ResourceVersion); err != nil {
		return err
	}
	if opts.Predicate.Label == nil || opts.Predicate.Field == nil {
		// etcd3 and the cacher call methods on both selectors.
		panic("nil label or field selector is not supported, use storage.Everything to match everything")
	}
	if opts.Predicate.Limit != 0 || opts.Predicate.Continue != "" {
		panic("pagination (limit, continue) is not supported")
	}
	if !opts.Predicate.Empty() && opts.Predicate.GetAttrs == nil {
		panic("selectors without GetAttrs are not supported")
	}
	if opts.RecordTimestamps {
		panic("recordTimestamps is not supported, it wraps objects in storage-internal types")
	}
	if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
		panic("initial events are not supported, set sendInitialEvents=false for resourceVersion \"\" or \"0\"")
	}
	return nil
}

func validateMaybeLastEventError(events []watch.Event) error {
	for i, ev := range events {
		if ev.Type != watch.Error {
			continue
		}
		if _, ok := ev.Object.(*metav1.Status); !ok {
			return fmt.Errorf("expected *metav1.Status in watch.Error event, got %T", ev.Object)
		}
		if i != len(events)-1 {
			return fmt.Errorf("watch.Error at index %d is not the last event (total %d)", i, len(events))
		}
	}
	return nil
}

func (v WatchValidator) validateReliable(request WatchRequest, response WatchResponse) error {
	rangeRV, err := watchRevisionRange(v.versioner, request, response.Events)
	if err != nil {
		return err
	}
	if rangeRV == nil {
		// No RV marking events to determine range of revisions.
		// As watch is eventually consistent, empty watch is correct.
		return nil
	}
	expected, err := v.replay.Events(request, rangeRV)
	if err != nil {
		return err
	}
	expectedRefs, err := v.toEventReference(expected)
	if err != nil {
		return err
	}
	gotEvents := filterOutBookmarksAndErrors(response.Events)
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
		if ev.Type == watch.Error {
			continue
		}
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
		rv, err := objectRV(event.Object, v.replay.versioner)
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

func (c Change) toWatchEvent(versioner storage.Versioner, pred storage.SelectionPredicate) (*watch.Event, error) {
	if c.Object == nil && c.PrevObject == nil {
		panic(fmt.Sprintf("change at resource version %d has neither Object nor PrevObject", c.ResourceVersion))
	}
	var curMatches, prevMatches bool
	var err error
	if c.Object != nil {
		curMatches, err = pred.Matches(c.Object)
		if err != nil {
			return nil, err
		}
	}
	if c.PrevObject != nil {
		prevMatches, err = pred.Matches(c.PrevObject)
		if err != nil {
			return nil, err
		}
	}
	switch {
	case curMatches && prevMatches:
		return &watch.Event{Type: watch.Modified, Object: c.Object}, nil
	case curMatches:
		return &watch.Event{Type: watch.Added, Object: c.Object}, nil
	case prevMatches:
		// The watcher gets the last state it matched, at the RV of the write
		// that made it stop matching.
		prev := c.PrevObject.DeepCopyObject()
		if err := versioner.UpdateObject(prev, c.ResourceVersion); err != nil {
			return nil, err
		}
		return &watch.Event{Type: watch.Deleted, Object: prev}, nil
	}
	return nil, nil
}

func watchRevisionRange(versioner storage.Versioner, request WatchRequest, events []watch.Event) (*ResourceVersionRange, error) {
	eventsRVs, err := eventResourceVersionRange(events, versioner)
	if err != nil {
		return nil, err
	}
	// No events to mark RV.
	if eventsRVs == nil {
		return nil, nil
	}
	requestedRV, err := versioner.ParseResourceVersion(request.Options.ResourceVersion)
	if err != nil {
		return nil, err
	}
	switch request.Options.ResourceVersionMatch {
	case metav1.ResourceVersionMatchNotOlderThan:
		if requestedRV > 0 {
			eventsRVs.Min = max(eventsRVs.Min, requestedRV+1)
		}
	case metav1.ResourceVersionMatchExact:
		if requestedRV == 0 {
			return nil, fmt.Errorf("exact resource version match is not supported for resource version %q", request.Options.ResourceVersion)
		}
		eventsRVs.Min = requestedRV + 1
	case "":
		if requestedRV > 0 {
			// exact match
			eventsRVs.Min = requestedRV + 1
		}
	default:
		return nil, fmt.Errorf("unknown resource version match: %s", request.Options.ResourceVersionMatch)
	}
	return eventsRVs, nil
}

func eventResourceVersionRange(events []watch.Event, versioner storage.Versioner) (*ResourceVersionRange, error) {
	minRV := uint64(math.MaxUint64)
	maxRV := uint64(0)
	for _, ev := range events {
		if ev.Type == watch.Error {
			continue
		}
		rv, err := objectRV(ev.Object, versioner)
		if err != nil {
			return nil, err
		}
		switch ev.Type {
		case watch.Added, watch.Deleted, watch.Modified:
			minRV = min(minRV, rv)
			maxRV = max(maxRV, rv+1)
		case watch.Bookmark:
			minRV = min(minRV, rv+1)
			maxRV = max(maxRV, rv)
		default:
			return nil, fmt.Errorf("unknown watch event type: %v", ev.Type)
		}
	}
	if minRV == uint64(math.MaxUint64) || maxRV == uint64(0) {
		return nil, nil // no events with RV
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

func filterOutBookmarksAndErrors(events []watch.Event) []watch.Event {
	filtered := make([]watch.Event, 0, len(events))
	for _, event := range events {
		if event.Type == watch.Bookmark || event.Type == watch.Error {
			continue
		}
		filtered = append(filtered, event)
	}
	return filtered
}
