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
	"context"

	eventsv1 "k8s.io/api/events/v1"
	"k8s.io/apimachinery/pkg/runtime"
	internalevents "k8s.io/client-go/tools/internal/events"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
)

type EventRecorder = internalevents.EventRecorder
type EventRecorderLogger = internalevents.EventRecorderLogger

// EventBroadcaster knows how to receive events and send them to any EventSink, watcher, or log.
type EventBroadcaster interface {
	// StartRecordingToSink starts sending events received from the specified eventBroadcaster.
	// Deprecated: use StartRecordingToSinkWithContext instead.
	StartRecordingToSink(stopCh <-chan struct{})

	// StartRecordingToSink starts sending events received from the specified eventBroadcaster.
	StartRecordingToSinkWithContext(ctx context.Context) error

	// NewRecorder returns an EventRecorder that can be used to send events to this EventBroadcaster
	// with the event source set to the given event source.
	NewRecorder(scheme *runtime.Scheme, reportingController string) EventRecorderLogger

	// StartEventWatcher enables you to watch for emitted events without usage
	// of StartRecordingToSink. This lets you also process events in a custom way (e.g. in tests).
	// NOTE: events received on your eventHandler should be copied before being used.
	// TODO: figure out if this can be removed.
	StartEventWatcher(eventHandler func(event runtime.Object)) (func(), error)

	// StartStructuredLogging starts sending events received from this EventBroadcaster to the structured
	// logging function. The return value can be ignored or used to stop recording, if desired.
	// Deprecated: use StartLogging instead.
	StartStructuredLogging(verbosity klog.Level) func()

	// StartLogging starts sending events received from this EventBroadcaster to the structured logger.
	// To adjust verbosity, use the logger's V method (i.e. pass `logger.V(3)` instead of `logger`).
	// The returned function can be ignored or used to stop recording, if desired.
	StartLogging(logger klog.Logger) (func(), error)

	// Shutdown shuts down the broadcaster
	Shutdown()
}

// EventSink knows how to store events (client-go implements it.)
// EventSink must respect the namespace that will be embedded in 'event'.
// It is assumed that EventSink will return the same sorts of errors as
// client-go's REST client.
type EventSink interface {
	Create(ctx context.Context, event *eventsv1.Event) (*eventsv1.Event, error)
	Update(ctx context.Context, event *eventsv1.Event) (*eventsv1.Event, error)
	Patch(ctx context.Context, oldEvent *eventsv1.Event, data []byte) (*eventsv1.Event, error)
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
	NewRecorder(name string) EventRecorderLogger

	// DeprecatedNewLegacyRecorder creates a legacy Event Recorder with specific name.
	DeprecatedNewLegacyRecorder(name string) record.EventRecorderLogger

	// Shutdown shuts down the broadcaster.
	Shutdown()
}
