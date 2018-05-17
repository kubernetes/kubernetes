/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"fmt"

	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/watch"
)

// WatcherFunc is a function that is responsible for creating a watcher starting at sinceResourceVersion.
type WatcherFunc func(sinceResourceVersion string) (watch.Interface, error)

// resourceVersionGetter is an interface used to get resource version from events.
// We can't reuse an interface from meta otherwise it would be a cyclic dependency and we need just this one method
type resourceVersionGetter interface {
	GetResourceVersion() string
}

// RetryWatcher will make sure that in case the underlying watcher is closed (e.g. due to API timeout or etcd timeout)
// it will get restarted from the last point without the consumer even knowing about it.
// RetryWatcher does that by inspecting events and keeping track of resourceVersion.
// Especially useful when using watch.Until where premature termination is causing troubles and flakes.
// Please note that this is not resilient to ETCD cache not having the resource version anymore - you would need to
// use Informers for that.
type RetryWatcher struct {
	lastResourceVersion string
	watcherFunc         WatcherFunc
	resultChan          chan watch.Event
	stopChan            chan struct{}
	doneChan            chan struct{}
}

// NewRetryWatcher creates a new RetryWatcher.
// It will make sure that watcher gets restarted in case of recoverable errors.
// The initialResourceVersion will be given to watchFunc when first called.
func NewRetryWatcher(initialResourceVersion string, watcherFunc WatcherFunc) *RetryWatcher {
	rw := &RetryWatcher{
		lastResourceVersion: initialResourceVersion,
		watcherFunc:         watcherFunc,
		stopChan:            make(chan struct{}),
		doneChan:            make(chan struct{}),
		resultChan:          make(chan watch.Event, 0),
	}

	go rw.receive()
	return rw
}

func (rw *RetryWatcher) send(event watch.Event) bool {
	// Writing to an unbuffered channel is blocking and we need to check if we need to be able to stop while doing so!
	select {
	case rw.resultChan <- event:
		return true
	case <-rw.stopChan:
		return false
	}
}

// doReceive returns true when it is done, false otherwise
func (rw *RetryWatcher) doReceive() bool {
	watcher, err := rw.watcherFunc(rw.lastResourceVersion)
	if err != nil {
		_ = rw.send(watch.Event{
			Type:   watch.Error,
			Object: apierrors.NewInternalError(fmt.Errorf("RetryWatcher: watcherFunc failed: %v", err)).Status(),
		})
		// Stop the watcher
		return true
	}
	ch := watcher.ResultChan()
	defer watcher.Stop()

	for {
		select {
		case <-rw.stopChan:
			glog.V(5).Info("Stopping RetryWatcher.")
			return true
		case event, ok := <-ch:
			if !ok {
				glog.Warningf("RetryWatcher - getting event failed! Re-creating the watcher. Last RV: %s", rw.lastResourceVersion)
				return false
			}

			// We need to inspect the event and get ResourceVersion out of it
			switch event.Type {
			case watch.Added, watch.Modified, watch.Deleted:
				metaObject, ok := event.Object.(resourceVersionGetter)
				if !ok {
					_ = rw.send(watch.Event{
						Type:   watch.Error,
						Object: apierrors.NewInternalError(errors.New("__internal__: RetryWatcher: doesn't support resourceVersion")).Status(),
					})
					// We have to abort here because this might cause lastResourceVersion inconsistency by skipping a potential RV with valid data!
					return true
				}

				resourceVersion := metaObject.GetResourceVersion()
				if resourceVersion == "" {
					_ = rw.send(watch.Event{
						Type:   watch.Error,
						Object: apierrors.NewInternalError(fmt.Errorf("__internal__: RetryWatcher: object %#v doesn't support resourceVersion", event.Object)).Status(),
					})
					// We have to abort here because this might cause lastResourceVersion inconsistency by skipping a potential RV with valid data!
					return true
				}

				// All is fine; send the event and update lastResourceVersion
				ok = rw.send(event)
				if !ok {
					return true
				}
				rw.lastResourceVersion = resourceVersion

				continue

			case watch.Error:
				_ = rw.send(event)
				return true

			default:
				glog.Errorf("RetryWatcher failed to recognize Event type %q", event.Type)
				_ = rw.send(watch.Event{
					Type:   watch.Error,
					Object: apierrors.NewInternalError(fmt.Errorf("__internal__: RetryWatcher failed to recognize Event type %q", event.Type)).Status(),
				})
				// We are unable to restart the watch and have to stop the loop or this might cause lastResourceVersion inconsistency by skipping a potential RV with valid data!
				return true
			}
		}
	}
}

// receive reads the result from a watcher, restarting it if necessary.
func (rw *RetryWatcher) receive() {
	defer close(rw.doneChan)

	for {
		select {
		case <-rw.stopChan:
			glog.V(5).Info("Stopping RetryWatcher.")
			return
		default:
			done := rw.doReceive()
			if done {
				return
			}
		}
	}
}

// ResultChan implements Interface.
func (rw *RetryWatcher) ResultChan() <-chan watch.Event {
	return rw.resultChan
}

// Stop implements Interface.
func (rw *RetryWatcher) Stop() {
	close(rw.stopChan)
}

// Done allows the caller to be notified when Retry watcher stops.
func (rw *RetryWatcher) Done() <-chan struct{} {
	return rw.doneChan
}
