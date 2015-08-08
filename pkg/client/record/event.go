/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package record

import (
	"fmt"
	"math/rand"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const maxTriesPerEvent = 12

var sleepDuration = 10 * time.Second

const maxQueuedEvents = 1000

// EventSink knows how to store events (client.Client implements it.)
// EventSink must respect the namespace that will be embedded in 'event'.
// It is assumed that EventSink will return the same sorts of errors as
// pkg/client's REST client.
type EventSink interface {
	Create(event *api.Event) (*api.Event, error)
	Update(event *api.Event) (*api.Event, error)
}

// EventRecorder knows how to record events on behalf of an EventSource.
type EventRecorder interface {
	// Event constructs an event from the given information and puts it in the queue for sending.
	// 'object' is the object this event is about. Event will make a reference-- or you may also
	// pass a reference to the object directly.
	// 'reason' is the reason this event is generated. 'reason' should be short and unique; it will
	// be used to automate handling of events, so imagine people writing switch statements to
	// handle them. You want to make that easy.
	// 'message' is intended to be human readable.
	//
	// The resulting event will be created in the same namespace as the reference object.
	Event(object runtime.Object, reason, message string)

	// Eventf is just like Event, but with Sprintf for the message field.
	Eventf(object runtime.Object, reason, messageFmt string, args ...interface{})

	// PastEventf is just like Eventf, but with an option to specify the event's 'timestamp' field.
	PastEventf(object runtime.Object, timestamp util.Time, reason, messageFmt string, args ...interface{})
}

// EventBroadcaster knows how to receive events and send them to any EventSink, watcher, or log.
type EventBroadcaster interface {
	// StartEventWatcher starts sending events received from this EventBroadcaster to the given
	// event handler function. The return value can be ignored or used to stop recording, if
	// desired.
	StartEventWatcher(eventHandler func(*api.Event)) watch.Interface

	// StartRecordingToSink starts sending events received from this EventBroadcaster to the given
	// sink. The return value can be ignored or used to stop recording, if desired.
	StartRecordingToSink(sink EventSink) watch.Interface

	// StartLogging starts sending events received from this EventBroadcaster to the given logging
	// function. The return value can be ignored or used to stop recording, if desired.
	StartLogging(logf func(format string, args ...interface{})) watch.Interface

	// NewRecorder returns an EventRecorder that can be used to send events to this EventBroadcaster
	// with the event source set to the given event source.
	NewRecorder(source api.EventSource) EventRecorder
}

// Creates a new event broadcaster.
func NewBroadcaster() EventBroadcaster {
	return &eventBroadcasterImpl{watch.NewBroadcaster(maxQueuedEvents, watch.DropIfChannelFull)}
}

type eventBroadcasterImpl struct {
	*watch.Broadcaster
}

// StartRecordingToSink starts sending events received from the specified eventBroadcaster to the given sink.
// The return value can be ignored or used to stop recording, if desired.
// TODO: make me an object with parameterizable queue length and retry interval
func (eventBroadcaster *eventBroadcasterImpl) StartRecordingToSink(sink EventSink) watch.Interface {
	// The default math/rand package functions aren't thread safe, so create a
	// new Rand object for each StartRecording call.
	randGen := rand.New(rand.NewSource(time.Now().UnixNano()))
	return eventBroadcaster.StartEventWatcher(
		func(event *api.Event) {
			// Make a copy before modification, because there could be multiple listeners.
			// Events are safe to copy like this.
			eventCopy := *event
			event = &eventCopy

			previousEvent := getEvent(event)
			updateExistingEvent := previousEvent.Count > 0
			if updateExistingEvent {
				event.Count = previousEvent.Count + 1
				event.FirstTimestamp = previousEvent.FirstTimestamp
				event.Name = previousEvent.Name
				event.ResourceVersion = previousEvent.ResourceVersion
			}

			tries := 0
			for {
				if recordEvent(sink, event, updateExistingEvent) {
					break
				}
				tries++
				if tries >= maxTriesPerEvent {
					glog.Errorf("Unable to write event '%#v' (retry limit exceeded!)", event)
					break
				}
				// Randomize the first sleep so that various clients won't all be
				// synced up if the master goes down.
				if tries == 1 {
					time.Sleep(time.Duration(float64(sleepDuration) * randGen.Float64()))
				} else {
					time.Sleep(sleepDuration)
				}
			}
		})
}

func isKeyNotFoundError(err error) bool {
	statusErr, _ := err.(*errors.StatusError)
	// At the moment the server is returning 500 instead of a more specific
	// error. When changing this remember that it should be backward compatible
	// with old api servers that may be still returning 500.
	if statusErr != nil && statusErr.Status().Code == 500 {
		return true
	}
	return false
}

// recordEvent attempts to write event to a sink. It returns true if the event
// was successfully recorded or discarded, false if it should be retried.
// If updateExistingEvent is false, it creates a new event, otherwise it updates
// existing event.
func recordEvent(sink EventSink, event *api.Event, updateExistingEvent bool) bool {
	var newEvent *api.Event
	var err error
	if updateExistingEvent {
		newEvent, err = sink.Update(event)
	}
	// Update can fail because the event may have been removed and it no longer exists.
	if !updateExistingEvent || (updateExistingEvent && isKeyNotFoundError(err)) {
		// Making sure that ResourceVersion is empty on creation
		event.ResourceVersion = ""
		newEvent, err = sink.Create(event)
	}
	if err == nil {
		addOrUpdateEvent(newEvent)
		return true
	}

	// If we can't contact the server, then hold everything while we keep trying.
	// Otherwise, something about the event is malformed and we should abandon it.
	switch err.(type) {
	case *client.RequestConstructionError:
		// We will construct the request the same next time, so don't keep trying.
		glog.Errorf("Unable to construct event '%#v': '%v' (will not retry!)", event, err)
		return true
	case *errors.StatusError:
		if errors.IsAlreadyExists(err) {
			glog.V(5).Infof("Server rejected event '%#v': '%v' (will not retry!)", event, err)
		} else {
			glog.Errorf("Server rejected event '%#v': '%v' (will not retry!)", event, err)
		}
		return true
	case *errors.UnexpectedObjectError:
		// We don't expect this; it implies the server's response didn't match a
		// known pattern. Go ahead and retry.
	default:
		// This case includes actual http transport errors. Go ahead and retry.
	}
	glog.Errorf("Unable to write event: '%v' (may retry after sleeping)", err)
	return false
}

// StartLogging starts sending events received from this EventBroadcaster to the given logging function.
// The return value can be ignored or used to stop recording, if desired.
func (eventBroadcaster *eventBroadcasterImpl) StartLogging(logf func(format string, args ...interface{})) watch.Interface {
	return eventBroadcaster.StartEventWatcher(
		func(e *api.Event) {
			logf("Event(%#v): reason: '%v' %v", e.InvolvedObject, e.Reason, e.Message)
		})
}

// StartEventWatcher starts sending events received from this EventBroadcaster to the given event handler function.
// The return value can be ignored or used to stop recording, if desired.
func (eventBroadcaster *eventBroadcasterImpl) StartEventWatcher(eventHandler func(*api.Event)) watch.Interface {
	watcher := eventBroadcaster.Watch()
	go func() {
		defer util.HandleCrash()
		for {
			watchEvent, open := <-watcher.ResultChan()
			if !open {
				return
			}
			event, ok := watchEvent.Object.(*api.Event)
			if !ok {
				// This is all local, so there's no reason this should
				// ever happen.
				continue
			}
			eventHandler(event)
		}
	}()
	return watcher
}

// NewRecorder returns an EventRecorder that records events with the given event source.
func (eventBroadcaster *eventBroadcasterImpl) NewRecorder(source api.EventSource) EventRecorder {
	return &recorderImpl{source, eventBroadcaster.Broadcaster}
}

type recorderImpl struct {
	source api.EventSource
	*watch.Broadcaster
}

func (recorder *recorderImpl) generateEvent(object runtime.Object, timestamp util.Time, reason, message string) {
	ref, err := api.GetReference(object)
	if err != nil {
		glog.Errorf("Could not construct reference to: '%#v' due to: '%v'. Will not report event: '%v' '%v'", object, err, reason, message)
		return
	}

	event := makeEvent(ref, reason, message)
	event.Source = recorder.source

	recorder.Action(watch.Added, event)
}

func (recorder *recorderImpl) Event(object runtime.Object, reason, message string) {
	recorder.generateEvent(object, util.Now(), reason, message)
}

func (recorder *recorderImpl) Eventf(object runtime.Object, reason, messageFmt string, args ...interface{}) {
	recorder.Event(object, reason, fmt.Sprintf(messageFmt, args...))
}

func (recorder *recorderImpl) PastEventf(object runtime.Object, timestamp util.Time, reason, messageFmt string, args ...interface{}) {
	recorder.generateEvent(object, timestamp, reason, fmt.Sprintf(messageFmt, args...))
}

func makeEvent(ref *api.ObjectReference, reason, message string) *api.Event {
	t := util.Now()
	namespace := ref.Namespace
	if namespace == "" {
		namespace = api.NamespaceDefault
	}
	return &api.Event{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: *ref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
	}
}
