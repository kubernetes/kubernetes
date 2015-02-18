/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const maxTriesPerEvent = 12

var sleepDuration = 10 * time.Second

// EventRecorder knows how to store events (client.Client implements it.)
// EventRecorder must respect the namespace that will be embedded in 'event'.
// It is assumed that EventRecorder will return the same sorts of errors as
// pkg/client's REST client.
type EventRecorder interface {
	Create(event *api.Event) (*api.Event, error)
	Update(event *api.Event) (*api.Event, error)
}

// StartRecording starts sending events to recorder. Call once while initializing
// your binary. Subsequent calls will be ignored. The return value can be ignored
// or used to stop recording, if desired.
// TODO: make me an object with parameterizable queue length and retry interval
func StartRecording(recorder EventRecorder, source api.EventSource) watch.Interface {
	// The default math/rand package functions aren't thread safe, so create a
	// new Rand object for each StartRecording call.
	randGen := rand.New(rand.NewSource(time.Now().UnixNano()))
	return GetEvents(func(event *api.Event) {
		// Make a copy before modification, because there could be multiple listeners.
		// Events are safe to copy like this.
		eventCopy := *event
		event = &eventCopy
		event.Source = source

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
			if recordEvent(recorder, event, updateExistingEvent) {
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

// recordEvent attempts to write event to recorder. It returns true if the event
// was successfully recorded or discarded, false if it should be retried.
// If updateExistingEvent is false, it creates a new event, otherwise it updates
// existing event.
func recordEvent(recorder EventRecorder, event *api.Event, updateExistingEvent bool) bool {
	var newEvent *api.Event
	var err error
	if updateExistingEvent {
		newEvent, err = recorder.Update(event)
	} else {
		newEvent, err = recorder.Create(event)
	}
	if err == nil {
		addOrUpdateEvent(newEvent)
		return true
	}

	// If we can't contact the server, then hold everything while we keep trying.
	// Otherwise, something about the event is malformed and we should abandon it.
	giveUp := false
	switch err.(type) {
	case *client.RequestConstructionError:
		// We will construct the request the same next time, so don't keep trying.
		giveUp = true
	case *errors.StatusError:
		// This indicates that the server understood and rejected our request.
		giveUp = true
	case *errors.UnexpectedObjectError:
		// We don't expect this; it implies the server's response didn't match a
		// known pattern. Go ahead and retry.
	default:
		// This case includes actual http transport errors. Go ahead and retry.
	}
	if giveUp {
		glog.Errorf("Unable to write event '%#v': '%v' (will not retry!)", event, err)
		return true
	}
	glog.Errorf("Unable to write event: '%v' (may retry after sleeping)", err)
	return false
}

// StartLogging just logs local events, using the given logging function. The
// return value can be ignored or used to stop logging, if desired.
func StartLogging(logf func(format string, args ...interface{})) watch.Interface {
	return GetEvents(func(e *api.Event) {
		logf("Event(%#v): reason: '%v' %v", e.InvolvedObject, e.Reason, e.Message)
	})
}

// GetEvents lets you see *local* events. Convenience function for testing. The
// return value can be ignored or used to stop logging, if desired.
func GetEvents(f func(*api.Event)) watch.Interface {
	w := events.Watch()
	go func() {
		defer util.HandleCrash()
		for {
			watchEvent, open := <-w.ResultChan()
			if !open {
				return
			}
			event, ok := watchEvent.Object.(*api.Event)
			if !ok {
				// This is all local, so there's no reason this should
				// ever happen.
				continue
			}
			f(event)
		}
	}()
	return w
}

const maxQueuedEvents = 1000

var events = watch.NewBroadcaster(maxQueuedEvents, watch.DropIfChannelFull)

// Event constructs an event from the given information and puts it in the queue for sending.
// 'object' is the object this event is about. Event will make a reference-- or you may also
// pass a reference to the object directly.
// 'reason' is the reason this event is generated. 'reason' should be short and unique; it will
// be used to automate handling of events, so imagine people writing switch statements to
// handle them. You want to make that easy.
// 'message' is intended to be human readable.
//
// The resulting event will be created in the same namespace as the reference object.
func Event(object runtime.Object, reason, message string) {
	ref, err := api.GetReference(object)
	if err != nil {
		glog.Errorf("Could not construct reference to: '%#v' due to: '%v'. Will not report event: '%v' '%v'", object, err, reason, message)
		return
	}
	t := util.Now()

	e := &api.Event{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: ref.Namespace,
		},
		InvolvedObject: *ref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
	}

	events.Action(watch.Added, e)
}

// Eventf is just like Event, but with Sprintf for the message field.
func Eventf(object runtime.Object, reason, messageFmt string, args ...interface{}) {
	Event(object, reason, fmt.Sprintf(messageFmt, args...))
}
