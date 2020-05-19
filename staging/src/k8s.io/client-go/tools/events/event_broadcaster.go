/*
Copyright 2019 The Kubernetes Authors.

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

package events

import (
	"os"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"

	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	typedv1beta1 "k8s.io/client-go/kubernetes/typed/events/v1beta1"
	"k8s.io/client-go/tools/record/util"
	"k8s.io/klog/v2"
)

const (
	maxTriesPerEvent = 12
	finishTime       = 6 * time.Minute
	refreshTime      = 30 * time.Minute
	maxQueuedEvents  = 1000
)

var defaultSleepDuration = 10 * time.Second

// TODO: validate impact of copying and investigate hashing
type eventKey struct {
	action              string
	reason              string
	reportingController string
	regarding           corev1.ObjectReference
	related             corev1.ObjectReference
}

type eventBroadcasterImpl struct {
	*watch.Broadcaster
	mu            sync.Mutex
	eventCache    map[eventKey]*v1beta1.Event
	sleepDuration time.Duration
	sink          EventSink
}

// EventSinkImpl wraps EventInterface to implement EventSink.
// TODO: this makes it easier for testing purpose and masks the logic of performing API calls.
// Note that rollbacking to raw clientset should also be transparent.
type EventSinkImpl struct {
	Interface typedv1beta1.EventInterface
}

// Create is the same as CreateWithEventNamespace of the EventExpansion
func (e *EventSinkImpl) Create(event *v1beta1.Event) (*v1beta1.Event, error) {
	return e.Interface.CreateWithEventNamespace(event)
}

// Update is the same as UpdateithEventNamespace of the EventExpansion
func (e *EventSinkImpl) Update(event *v1beta1.Event) (*v1beta1.Event, error) {
	return e.Interface.UpdateWithEventNamespace(event)
}

// Patch is the same as PatchWithEventNamespace of the EventExpansion
func (e *EventSinkImpl) Patch(event *v1beta1.Event, data []byte) (*v1beta1.Event, error) {
	return e.Interface.PatchWithEventNamespace(event, data)
}

// NewBroadcaster Creates a new event broadcaster.
func NewBroadcaster(sink EventSink) EventBroadcaster {
	return newBroadcaster(sink, defaultSleepDuration, map[eventKey]*v1beta1.Event{})
}

// NewBroadcasterForTest Creates a new event broadcaster for test purposes.
func newBroadcaster(sink EventSink, sleepDuration time.Duration, eventCache map[eventKey]*v1beta1.Event) EventBroadcaster {
	return &eventBroadcasterImpl{
		Broadcaster:   watch.NewBroadcaster(maxQueuedEvents, watch.DropIfChannelFull),
		eventCache:    eventCache,
		sleepDuration: sleepDuration,
		sink:          sink,
	}
}

func (e *eventBroadcasterImpl) Shutdown() {
	e.Broadcaster.Shutdown()
}

// refreshExistingEventSeries refresh events TTL
func (e *eventBroadcasterImpl) refreshExistingEventSeries() {
	// TODO: Investigate whether lock contention won't be a problem
	e.mu.Lock()
	defer e.mu.Unlock()
	for isomorphicKey, event := range e.eventCache {
		if event.Series != nil {
			if recordedEvent, retry := recordEvent(e.sink, event); !retry {
				if recordedEvent != nil {
					e.eventCache[isomorphicKey] = recordedEvent
				}
			}
		}
	}
}

// finishSeries checks if a series has ended and either:
// - write final count to the apiserver
// - delete a singleton event (i.e. series field is nil) from the cache
func (e *eventBroadcasterImpl) finishSeries() {
	// TODO: Investigate whether lock contention won't be a problem
	e.mu.Lock()
	defer e.mu.Unlock()
	for isomorphicKey, event := range e.eventCache {
		eventSerie := event.Series
		if eventSerie != nil {
			if eventSerie.LastObservedTime.Time.Before(time.Now().Add(-finishTime)) {
				if _, retry := recordEvent(e.sink, event); !retry {
					delete(e.eventCache, isomorphicKey)
				}
			}
		} else if event.EventTime.Time.Before(time.Now().Add(-finishTime)) {
			delete(e.eventCache, isomorphicKey)
		}
	}
}

// NewRecorder returns an EventRecorder that records events with the given event source.
func (e *eventBroadcasterImpl) NewRecorder(scheme *runtime.Scheme, reportingController string) EventRecorder {
	hostname, _ := os.Hostname()
	reportingInstance := reportingController + "-" + hostname
	return &recorderImpl{scheme, reportingController, reportingInstance, e.Broadcaster, clock.RealClock{}}
}

func (e *eventBroadcasterImpl) recordToSink(event *v1beta1.Event, clock clock.Clock) {
	// Make a copy before modification, because there could be multiple listeners.
	eventCopy := event.DeepCopy()
	go func() {
		evToRecord := func() *v1beta1.Event {
			e.mu.Lock()
			defer e.mu.Unlock()
			eventKey := getKey(eventCopy)
			isomorphicEvent, isIsomorphic := e.eventCache[eventKey]
			if isIsomorphic {
				if isomorphicEvent.Series != nil {
					isomorphicEvent.Series.Count++
					isomorphicEvent.Series.LastObservedTime = metav1.MicroTime{Time: clock.Now()}
					return nil
				}
				isomorphicEvent.Series = &v1beta1.EventSeries{
					Count:            1,
					LastObservedTime: metav1.MicroTime{Time: clock.Now()},
				}
				return isomorphicEvent
			}
			e.eventCache[eventKey] = eventCopy
			return eventCopy
		}()
		if evToRecord != nil {
			recordedEvent := e.attemptRecording(evToRecord)
			if recordedEvent != nil {
				recordedEventKey := getKey(recordedEvent)
				e.mu.Lock()
				defer e.mu.Unlock()
				e.eventCache[recordedEventKey] = recordedEvent
			}
		}
	}()
}

func (e *eventBroadcasterImpl) attemptRecording(event *v1beta1.Event) *v1beta1.Event {
	tries := 0
	for {
		if recordedEvent, retry := recordEvent(e.sink, event); !retry {
			return recordedEvent
		}
		tries++
		if tries >= maxTriesPerEvent {
			klog.Errorf("Unable to write event '%#v' (retry limit exceeded!)", event)
			return nil
		}
		// Randomize sleep so that various clients won't all be
		// synced up if the master goes down.
		time.Sleep(wait.Jitter(e.sleepDuration, 0.25))
	}
}

func recordEvent(sink EventSink, event *v1beta1.Event) (*v1beta1.Event, bool) {
	var newEvent *v1beta1.Event
	var err error
	isEventSeries := event.Series != nil
	if isEventSeries {
		patch, patchBytesErr := createPatchBytesForSeries(event)
		if patchBytesErr != nil {
			klog.Errorf("Unable to calculate diff, no merge is possible: %v", patchBytesErr)
			return nil, false
		}
		newEvent, err = sink.Patch(event, patch)
	}
	// Update can fail because the event may have been removed and it no longer exists.
	if !isEventSeries || (isEventSeries && util.IsKeyNotFoundError(err)) {
		// Making sure that ResourceVersion is empty on creation
		event.ResourceVersion = ""
		newEvent, err = sink.Create(event)
	}
	if err == nil {
		return newEvent, false
	}
	// If we can't contact the server, then hold everything while we keep trying.
	// Otherwise, something about the event is malformed and we should abandon it.
	switch err.(type) {
	case *restclient.RequestConstructionError:
		// We will construct the request the same next time, so don't keep trying.
		klog.Errorf("Unable to construct event '%#v': '%v' (will not retry!)", event, err)
		return nil, false
	case *errors.StatusError:
		if errors.IsAlreadyExists(err) {
			klog.V(5).Infof("Server rejected event '%#v': '%v' (will not retry!)", event, err)
		} else {
			klog.Errorf("Server rejected event '%#v': '%v' (will not retry!)", event, err)
		}
		return nil, false
	case *errors.UnexpectedObjectError:
		// We don't expect this; it implies the server's response didn't match a
		// known pattern. Go ahead and retry.
	default:
		// This case includes actual http transport errors. Go ahead and retry.
	}
	klog.Errorf("Unable to write event: '%v' (may retry after sleeping)", err)
	return nil, true
}

func createPatchBytesForSeries(event *v1beta1.Event) ([]byte, error) {
	oldEvent := event.DeepCopy()
	oldEvent.Series = nil
	oldData, err := json.Marshal(oldEvent)
	if err != nil {
		return nil, err
	}
	newData, err := json.Marshal(event)
	if err != nil {
		return nil, err
	}
	return strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1beta1.Event{})
}

func getKey(event *v1beta1.Event) eventKey {
	key := eventKey{
		action:              event.Action,
		reason:              event.Reason,
		reportingController: event.ReportingController,
		regarding:           event.Regarding,
	}
	if event.Related != nil {
		key.related = *event.Related
	}
	return key
}

// StartEventWatcher starts sending events received from this EventBroadcaster to the given event handler function.
// The return value is used to stop recording
func (e *eventBroadcasterImpl) StartEventWatcher(eventHandler func(event runtime.Object)) func() {
	watcher := e.Watch()
	go func() {
		defer utilruntime.HandleCrash()
		for {
			watchEvent, ok := <-watcher.ResultChan()
			if !ok {
				return
			}
			eventHandler(watchEvent.Object)
		}
	}()
	return watcher.Stop
}

// StartRecordingToSink starts sending events received from the specified eventBroadcaster to the given sink.
func (e *eventBroadcasterImpl) StartRecordingToSink(stopCh <-chan struct{}) {
	go wait.Until(func() {
		e.refreshExistingEventSeries()
	}, refreshTime, stopCh)
	go wait.Until(func() {
		e.finishSeries()
	}, finishTime, stopCh)
	eventHandler := func(obj runtime.Object) {
		event, ok := obj.(*v1beta1.Event)
		if !ok {
			klog.Errorf("unexpected type, expected v1beta1.Event")
			return
		}
		e.recordToSink(event, clock.RealClock{})
	}
	stopWatcher := e.StartEventWatcher(eventHandler)
	go func() {
		<-stopCh
		stopWatcher()
	}()
}
