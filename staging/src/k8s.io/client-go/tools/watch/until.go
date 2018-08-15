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
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
)

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

// ContextWithOptionalTimeout wraps context.WithTimeout and handles infinite timeouts expressed as 0 duration.
func ContextWithOptionalTimeout(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout < 0 {
		// This should be handled in validation
		glog.Errorf("Timeout for context shall not be negative!")
		timeout = 0
	}

	if timeout == 0 {
		return context.WithCancel(parent)
	}

	return context.WithTimeout(parent, timeout)
}
