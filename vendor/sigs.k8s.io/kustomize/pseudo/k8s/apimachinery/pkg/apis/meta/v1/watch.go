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

package v1

import (
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/watch"
)

// Event represents a single event to a watched resource.
//
// +protobuf=true
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime.Object
type WatchEvent struct {
	Type string `json:"type" protobuf:"bytes,1,opt,name=type"`

	// Object is:
	//  * If Type is Added or Modified: the new state of the object.
	//  * If Type is Deleted: the state of the object immediately before deletion.
	//  * If Type is Error: *Status is recommended; other types may make sense
	//    depending on context.
	Object runtime.RawExtension `json:"object" protobuf:"bytes,2,opt,name=object"`
}

func Convert_watch_Event_To_v1_WatchEvent(in *watch.Event, out *WatchEvent, s conversion.Scope) error {
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

func Convert_v1_InternalEvent_To_v1_WatchEvent(in *InternalEvent, out *WatchEvent, s conversion.Scope) error {
	return Convert_watch_Event_To_v1_WatchEvent((*watch.Event)(in), out, s)
}

func Convert_v1_WatchEvent_To_watch_Event(in *WatchEvent, out *watch.Event, s conversion.Scope) error {
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

func Convert_v1_WatchEvent_To_v1_InternalEvent(in *WatchEvent, out *InternalEvent, s conversion.Scope) error {
	return Convert_v1_WatchEvent_To_watch_Event(in, (*watch.Event)(out), s)
}

// InternalEvent makes watch.Event versioned
// +protobuf=false
type InternalEvent watch.Event

func (e *InternalEvent) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (e *WatchEvent) GetObjectKind() schema.ObjectKind    { return schema.EmptyObjectKind }
func (e *InternalEvent) DeepCopyObject() runtime.Object {
	if c := e.DeepCopy(); c != nil {
		return c
	} else {
		return nil
	}
}
