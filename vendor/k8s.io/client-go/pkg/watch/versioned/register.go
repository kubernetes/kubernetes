/*
Copyright 2015 The Kubernetes Authors.

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

package versioned

import (
	"k8s.io/client-go/pkg/api/unversioned"
	"k8s.io/client-go/pkg/conversion"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/watch"
)

// WatchEventKind is name reserved for serializing watch events.
const WatchEventKind = "WatchEvent"

// AddToGroupVersion registers the watch external and internal kinds with the scheme, and ensures the proper
// conversions are in place.
func AddToGroupVersion(scheme *runtime.Scheme, groupVersion unversioned.GroupVersion) {
	scheme.AddKnownTypeWithName(groupVersion.WithKind(WatchEventKind), &Event{})
	scheme.AddKnownTypeWithName(
		unversioned.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}.WithKind(WatchEventKind),
		&InternalEvent{},
	)
	scheme.AddConversionFuncs(
		Convert_versioned_Event_to_watch_Event,
		Convert_versioned_InternalEvent_to_versioned_Event,
		Convert_watch_Event_to_versioned_Event,
		Convert_versioned_Event_to_versioned_InternalEvent,
	)
}

func Convert_watch_Event_to_versioned_Event(in *watch.Event, out *Event, s conversion.Scope) error {
	out.Type = string(in.Type)
	switch t := in.Object.(type) {
	case *runtime.Unknown:
		// TODO: handle other fields on Unknown and detect type
		out.Object.Raw = t.Raw
	case nil:
	default:
		out.Object.Object = in.Object
	}
	return nil
}

func Convert_versioned_InternalEvent_to_versioned_Event(in *InternalEvent, out *Event, s conversion.Scope) error {
	return Convert_watch_Event_to_versioned_Event((*watch.Event)(in), out, s)
}

func Convert_versioned_Event_to_watch_Event(in *Event, out *watch.Event, s conversion.Scope) error {
	out.Type = watch.EventType(in.Type)
	if in.Object.Object != nil {
		out.Object = in.Object.Object
	} else if in.Object.Raw != nil {
		// TODO: handle other fields on Unknown and detect type
		out.Object = &runtime.Unknown{
			Raw:         in.Object.Raw,
			ContentType: runtime.ContentTypeJSON,
		}
	}
	return nil
}

func Convert_versioned_Event_to_versioned_InternalEvent(in *Event, out *InternalEvent, s conversion.Scope) error {
	return Convert_versioned_Event_to_watch_Event(in, (*watch.Event)(out), s)
}

// InternalEvent makes watch.Event versioned
type InternalEvent watch.Event

func (e *InternalEvent) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }
func (e *Event) GetObjectKind() unversioned.ObjectKind         { return unversioned.EmptyObjectKind }
