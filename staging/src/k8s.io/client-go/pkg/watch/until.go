/*
Copyright 2016 The Kubernetes Authors.

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

package watch

import (
	"time"

	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/meta"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/util/wait"
)

// ConditionFunc returns true if the condition has been reached, false if it has not been reached yet,
// or an error if the condition cannot be checked and should terminate. In general, it is better to define
// level driven conditions over edge driven conditions (pod has ready=true, vs pod modified and ready changed
// from false to true).
type ConditionFunc func(event Event) (bool, error)

// Until reads items from the watch until each provided condition succeeds, and then returns the last watch
// encountered. The first condition that returns an error terminates the watch (and the event is also returned).
// If no event has been received, the returned event will be nil.
// Conditions are satisfied sequentially so as to provide a useful primitive for higher level composition.
// A zero timeout means to wait forever.
func Until(timeout time.Duration, watcher Interface, conditions ...ConditionFunc) (*Event, error) {
	ch := watcher.ResultChan()
	defer watcher.Stop()
	var after <-chan time.Time
	if timeout > 0 {
		after = time.After(timeout)
	} else {
		ch := make(chan time.Time)
		defer close(ch)
		after = ch
	}
	var lastEvent *Event
	for _, condition := range conditions {
		// check the next condition against the previous event and short circuit waiting for the next watch
		if lastEvent != nil {
			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				continue
			}
		}
	ConditionSucceeded:
		for {
			select {
			case event, ok := <-ch:
				if !ok {
					return lastEvent, wait.ErrWaitTimeout
				}
				lastEvent = &event

				// TODO: check for watch expired error and retry watch from latest point?
				done, err := condition(event)
				if err != nil {
					return lastEvent, err
				}
				if done {
					break ConditionSucceeded
				}

			case <-after:
				return lastEvent, wait.ErrWaitTimeout
			}
		}
	}
	return lastEvent, nil
}

// ListerWatcher is any object that knows how to perform an initial list and start a watch on a resource.
type ListerWatcher interface {
	// List should return a list type object; the Items field will be extracted, and the
	// ResourceVersion field will be used to start the watch in the right place.
	List(options api.ListOptions) (runtime.Object, error)
	// Watch should begin a watch at the specified version.
	Watch(options api.ListOptions) (Interface, error)
}

// TODO: check for watch expired error and retry watch from latest point?  Same issue exists for Until.
func ListWatchUntil(timeout time.Duration, lw ListerWatcher, conditions ...ConditionFunc) (*Event, error) {
	if len(conditions) == 0 {
		return nil, nil
	}

	list, err := lw.List(api.ListOptions{})
	if err != nil {
		return nil, err
	}
	initialItems, err := meta.ExtractList(list)
	if err != nil {
		return nil, err
	}

	// use the initial items as simulated "adds"
	var lastEvent *Event
	currIndex := 0
	passedConditions := 0
	for _, condition := range conditions {
		// check the next condition against the previous event and short circuit waiting for the next watch
		if lastEvent != nil {
			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				passedConditions = passedConditions + 1
				continue
			}
		}

	ConditionSucceeded:
		for currIndex < len(initialItems) {
			lastEvent = &Event{Type: Added, Object: initialItems[currIndex]}
			currIndex++

			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				passedConditions = passedConditions + 1
				break ConditionSucceeded
			}
		}
	}
	if passedConditions == len(conditions) {
		return lastEvent, nil
	}
	remainingConditions := conditions[passedConditions:]

	metaObj, err := meta.ListAccessor(list)
	if err != nil {
		return nil, err
	}
	currResourceVersion := metaObj.GetResourceVersion()

	watch, err := lw.Watch(api.ListOptions{ResourceVersion: currResourceVersion})
	if err != nil {
		return nil, err
	}

	return Until(timeout, watch, remainingConditions...)
}
