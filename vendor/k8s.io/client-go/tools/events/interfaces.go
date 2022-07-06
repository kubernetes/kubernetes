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
	eventsv1 "k8s.io/api/events/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
)

// EventRecorder knows how to record events on behalf of an EventSource.
type EventRecorder interface {
	// Eventf constructs an event from the given information and puts it in the queue for sending.
	// 'regarding' is the object this event is about. Event will make a reference-- or you may also
	// pass a reference to the object directly.
	// 'related' is the secondary object for more complex actions. E.g. when regarding object triggers
	// a creation or deletion of related object.
	// 'type' of this event, and can be one of Normal, Warning. New types could be added in future
	// 'reason' is the reason this event is generated. 'reason' should be short and unique; it
	// should be in UpperCamelCase format (starting with a capital letter). "reason" will be used
	// to automate handling of events, so imagine people writing switch statements to handle them.
	// You want to make that easy.
	// 'action' explains what happened with regarding/what action did the ReportingController
	// (ReportingController is a type of a Controller reporting an Event, e.g. k8s.io/node-controller, k8s.io/kubelet.)
	// take in regarding's name; it should be in UpperCamelCase format (starting with a capital letter).
	// 'note' is intended to be human readable.
	Eventf(regarding runtime.Object, related runtime.Object, eventtype, reason, action, note string, args ...interface{})
}

// EventBroadcaster knows how to receive events and send them to any EventSink, watcher, or log.
type EventBroadcaster interface {
	// StartRecordingToSink starts sending events received from the specified eventBroadcaster.
	StartRecordingToSink(stopCh <-chan struct{})

	// NewRecorder returns an EventRecorder that can be used to send events to this EventBroadcaster
	// with the event source set to the given event source.
	NewRecorder(scheme *runtime.Scheme, reportingController string) EventRecorder

	// StartEventWatcher enables you to watch for emitted events without usage
	// of StartRecordingToSink. This lets you also process events in a custom way (e.g. in tests).
	// NOTE: events received on your eventHandler should be copied before being used.
	// TODO: figure out if this can be removed.
	StartEventWatcher(eventHandler func(event runtime.Object)) func()

	// StartStructuredLogging starts sending events received from this EventBroadcaster to the structured
	// logging function. The return value can be ignored or used to stop recording, if desired.
	StartStructuredLogging(verbosity klog.Level) func()

	// Shutdown shuts down the broadcaster
	Shutdown()
}

// EventSink knows how to store events (client-go implements it.)
// EventSink must respect the namespace that will be embedded in 'event'.
// It is assumed that EventSink will return the same sorts of errors as
// client-go's REST client.
type EventSink interface {
	Create(event *eventsv1.Event) (*eventsv1.Event, error)
	Update(event *eventsv1.Event) (*eventsv1.Event, error)
	Patch(oldEvent *eventsv1.Event, data []byte) (*eventsv1.Event, error)
}

// EventBroadcasterAdapter is a auxiliary interface to simplify migration to
// the new events API. It is a wrapper around new and legacy broadcasters
// that smartly chooses which one to use.
//
// Deprecated: This interface will be removed once migration is completed.
type EventBroadcasterAdapter interface {
	// StartRecordingToSink starts sending events received from the specified eventBroadcaster.
	StartRecordingToSink(stopCh <-chan struct{})

	// NewRecorder creates a new Event Recorder with specified name.
	NewRecorder(name string) EventRecorder

	// DeprecatedNewLegacyRecorder creates a legacy Event Recorder with specific name.
	DeprecatedNewLegacyRecorder(name string) record.EventRecorder

	// Shutdown shuts down the broadcaster.
	Shutdown()
}
