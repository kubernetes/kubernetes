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

// Package internal is needed to break an import cycle: record.EventRecorderAdapter
// needs this interface definition to implement it, but event.NewEventBroadcasterAdapter
// needs record.NewBroadcaster. Therefore this interface cannot be in event/interfaces.go.
package internal

import (
	"k8s.io/apimachinery/pkg/runtime"
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

// EventRecorderLogger extends EventRecorder such that a logger can
// be set for methods in EventRecorder. Normally, those methods
// uses the global default logger to record errors and debug messages.
// If that is not desired, use WithLogger to provide a logger instance.
type EventRecorderLogger interface {
	EventRecorder

	// WithLogger replaces the context used for logging. This is a cheap call
	// and meant to be used for contextual logging:
	//    recorder := ...
	//    logger := klog.FromContext(ctx)
	//    recorder.WithLogger(logger).Eventf(...)
	WithLogger(logger klog.Logger) EventRecorderLogger
}
