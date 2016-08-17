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

// Package versioned contains the versioned types for watch. This is the first
// serialization version unless otherwise noted.
package versioned

import (
	"k8s.io/client-go/1.4/pkg/runtime"
)

// Event represents a single event to a watched resource.
//
// +protobuf=true
type Event struct {
	Type string `json:"type" protobuf:"bytes,1,opt,name=type"`

	// Object is:
	//  * If Type is Added or Modified: the new state of the object.
	//  * If Type is Deleted: the state of the object immediately before deletion.
	//  * If Type is Error: *api.Status is recommended; other types may make sense
	//    depending on context.
	Object runtime.RawExtension `json:"object" protobuf:"bytes,2,opt,name=object"`
}
