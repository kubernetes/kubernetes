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

package api

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// WatchEvent objects are streamed from the api server in response to a watch request.
// These are not API objects and are unversioned today.
type WatchEvent struct {
	// The type of the watch event; added, modified, or deleted.
	Type watch.EventType

	// For added or modified objects, this is the new object; for deleted objects,
	// it's the state of the object immediately prior to its deletion.
	Object EmbeddedObject
}

// watchSerialization defines the JSON wire equivalent of watch.Event
type watchSerialization struct {
	Type   watch.EventType
	Object json.RawMessage
}

// NewJSONWatcHEvent returns an object that will serialize to JSON and back
// to a WatchEvent.
func NewJSONWatchEvent(codec runtime.Codec, event watch.Event) (interface{}, error) {
	obj, ok := event.Object.(runtime.Object)
	if !ok {
		return nil, fmt.Errorf("The event object cannot be safely converted to JSON: %v", reflect.TypeOf(event.Object).Name())
	}
	data, err := codec.Encode(obj)
	if err != nil {
		return nil, err
	}
	return &watchSerialization{event.Type, json.RawMessage(data)}, nil
}
