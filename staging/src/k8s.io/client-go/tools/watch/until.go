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
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

// PreconditionFunc returns true if the condition has been reached, false if it has not been reached yet,
// or an error if the condition failed or detected an error state.
type PreconditionFunc func(store cache.Store) (bool, error)

// ConditionFunc returns true if the condition has been reached, false if it has not been reached yet,
// or an error if the condition cannot be checked and should terminate. In general, it is better to define
// level driven conditions over edge driven conditions (pod has ready=true, vs pod modified and ready changed
// from false to true).
type ConditionFunc func(event watch.Event) (bool, error)

// ErrWatchClosed is returned when the watch channel is closed before timeout in UntilWithoutRetry.
var ErrWatchClosed = errors.New("watch closed before UntilWithoutRetry timeout")

// UntilWithoutRetry reads items from the watch until each provided condition succeeds, and then returns the last watch
// encountered. The first condition that returns an error terminates the watch (and the event is also returned).
// If no event has been received, the returned event will be nil.
// Conditions are satisfied sequentially so as to provide a useful primitive for higher level composition.
// Waits until context deadline or until context is canceled.
//
// Warning: Unless you have a very specific use case (probably a special Watcher) don't use this function!!!
// Warning: This will fail e.g. on API timeouts and/or 'too old resource version' error.
// Warning: You are most probably looking for a function *Until* or *UntilWithSync* below,
// Warning: solving such issues.
// TODO: Consider making this function private to prevent misuse when the other occurrences in our codebase are gone.
func UntilWithoutRetry(ctx context.Context, watcher watch.Interface, conditions ...ConditionFunc) (*watch.Event, error) {
	ch := watcher.ResultChan()
	defer watcher.Stop()
	var lastEvent *watch.Event
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
					return lastEvent, ErrWatchClosed
				}
				lastEvent = &event

				done, err := condition(event)
				if err != nil {
					return lastEvent, err
				}
				if done {
					break ConditionSucceeded
				}

			case <-ctx.Done():
				return lastEvent, wait.ErrWaitTimeout
			}
		}
	}
	return lastEvent, nil
}

// UntilWithSync creates an informer from lw, optionally checks precondition when the store is synced,
// and watches the output until each provided condition succeeds, in a way that is identical
// to function UntilWithoutRetry. (See above.)
// UntilWithSync can deal with all errors like API timeout, lost connections and 'Resource version too old'.
// It is the only function that can recover from 'Resource version too old', Until and UntilWithoutRetry will
// just fail in that case. On the other hand it can't provide you with guarantees as strong as using simple
// Watch method with Until. It can skip some intermediate events in case of watch function failing but it will
// re-list to recover and you always get an event, if there has been a change, after recovery.
// Also with the current implementation based on DeltaFIFO, order of the events you receive is guaranteed only for
// particular object, not between more of them even it's the same resource.
// The most frequent usage would be a command that needs to watch the "state of the world" and should't fail, like:
// waiting for object reaching a state, "small" controllers, ...
func UntilWithSync(ctx context.Context, lw cache.ListerWatcher, objType runtime.Object, precondition PreconditionFunc, conditions ...ConditionFunc) (*watch.Event, error) {
	indexer, informer, watcher := NewIndexerInformerWatcher(lw, objType)
	// Proxy watcher can be stopped multiple times so it's fine to use defer here to cover alternative branches and
	// let UntilWithoutRetry to stop it
	defer watcher.Stop()

	if precondition != nil {
		if !cache.WaitForCacheSync(ctx.Done(), informer.HasSynced) {
			return nil, fmt.Errorf("UntilWithSync: unable to sync caches: %v", ctx.Err())
		}

		done, err := precondition(indexer)
		if err != nil {
			return nil, err
		}

		if done {
			return nil, nil
		}
	}

	return UntilWithoutRetry(ctx, watcher, conditions...)
}

// ContextWithOptionalTimeout wraps context.WithTimeout and handles infinite timeouts expressed as 0 duration.
func ContextWithOptionalTimeout(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout < 0 {
		// This should be handled in validation
		klog.Errorf("Timeout for context shall not be negative!")
		timeout = 0
	}

	if timeout == 0 {
		return context.WithCancel(parent)
	}

	return context.WithTimeout(parent, timeout)
}

// ListWatchUntil checks the provided conditions against the items returned by the list watcher, returning wait.ErrWaitTimeout
// if timeout is exceeded without all conditions returning true, or an error if an error occurs.
// TODO: check for watch expired error and retry watch from latest point?  Same issue exists for Until.
// TODO: remove when no longer used
//
// Deprecated: Use UntilWithSync instead.
func ListWatchUntil(timeout time.Duration, lw cache.ListerWatcher, conditions ...ConditionFunc) (*watch.Event, error) {
	if len(conditions) == 0 {
		return nil, nil
	}

	list, err := lw.List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	initialItems, err := meta.ExtractList(list)
	if err != nil {
		return nil, err
	}

	// use the initial items as simulated "adds"
	var lastEvent *watch.Event
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
			lastEvent = &watch.Event{Type: watch.Added, Object: initialItems[currIndex]}
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

	watchInterface, err := lw.Watch(metav1.ListOptions{ResourceVersion: currResourceVersion})
	if err != nil {
		return nil, err
	}

	ctx, cancel := ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	evt, err := UntilWithoutRetry(ctx, watchInterface, remainingConditions...)
	if err == ErrWatchClosed {
		// present a consistent error interface to callers
		err = wait.ErrWaitTimeout
	}
	return evt, err
}
