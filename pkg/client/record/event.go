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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// EventRecorder knows how to store events (client.Client implements it.)
// EventRecorder must respect the namespace that will be embedded in 'event'.
type EventRecorder interface {
	Create(event *api.Event) (*api.Event, error)
}

// StartRecording starts sending events to recorder. Call once while initializing
// your binary. Subsequent calls will be ignored. The return value can be ignored
// or used to stop recording, if desired.
func StartRecording(recorder EventRecorder, sourceName string) watch.Interface {
	return GetEvents(func(event *api.Event) {
		// Make a copy before modification, because there could be multiple listeners.
		// Events are safe to copy like this.
		eventCopy := *event
		event = &eventCopy
		event.Source = sourceName
		for {
			_, err := recorder.Create(event)
			if err == nil {
				break
			}
			glog.Errorf("Sleeping: Unable to write event: %v", err)
			time.Sleep(10 * time.Second)
		}
	})
}

// StartLogging just logs local events, using the given logging function. The
// return value can be ignored or used to stop logging, if desired.
func StartLogging(logf func(format string, args ...interface{})) watch.Interface {
	return GetEvents(func(e *api.Event) {
		logf("Event(%#v): status: '%v', reason: '%v' %v", e.InvolvedObject, e.Status, e.Reason, e.Message)
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

const queueLen = 1000

var events = watch.NewMux(queueLen)

// Event constructs an event from the given information and puts it in the queue for sending.
// 'object' is the object this event is about. Event will make a reference-- or you may also
// pass a reference to the object directly.
// 'status' is the new status of the object. 'reason' is the reason it now has this status.
// Both 'status' and 'reason' should be short and unique; they will be used to automate
// handling of events, so imagine people writing switch statements to handle them. You want to
// make that easy.
// 'message' is intended to be human readable.
//
// The resulting event will be created in the same namespace as the reference object.
func Event(object runtime.Object, status, reason, message string) {
	ref, err := api.GetReference(object)
	if err != nil {
		glog.Errorf("Could not construct reference to: '%#v' due to: '%v'. Will not report event: '%v' '%v' '%v'", object, err, status, reason, message)
		return
	}
	t := util.Now()

	e := &api.Event{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: ref.Namespace,
		},
		InvolvedObject: *ref,
		Status:         status,
		Reason:         reason,
		Message:        message,
		Timestamp:      t,
	}

	events.Action(watch.Added, e)
}

// Eventf is just like Event, but with Sprintf for the message field.
func Eventf(object runtime.Object, status, reason, messageFmt string, args ...interface{}) {
	Event(object, status, reason, fmt.Sprintf(messageFmt, args...))
}
