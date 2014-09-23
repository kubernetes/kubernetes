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

package json

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// watchEvent objects are streamed from the api server in response to a watch request.
// These are not API objects and are unversioned today.
type watchEvent struct {
	// The type of the watch event; added, modified, or deleted.
	Type watch.EventType `json:"type,omitempty" yaml:"type,omitempty"`

	// For added or modified objects, this is the new object; for deleted objects,
	// it's the state of the object immediately prior to its deletion.
	// For errors, it's an api.Status.
	Object runtime.RawExtension `json:"object,omitempty" yaml:"object,omitempty"`
}

// Object converts a watch.Event into an appropriately serializable JSON object
func Object(codec runtime.Codec, event *watch.Event) (interface{}, error) {
	obj, ok := event.Object.(runtime.Object)
	if !ok {
		return nil, fmt.Errorf("The event object cannot be safely converted to JSON: %v", reflect.TypeOf(event.Object).Name())
	}
	data, err := codec.Encode(obj)
	if err != nil {
		return nil, err
	}
	return &watchEvent{event.Type, runtime.RawExtension{json.RawMessage(data)}}, nil
}
