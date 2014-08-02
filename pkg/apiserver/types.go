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

package apiserver

import (
	api "github.com/GoogleCloudPlatform/kubernetes/pkg/api/internal"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Status is a return value for calls that don't return other objects.
// Arguably, this could go in apiserver, but I'm including it here so clients needn't
// import both.
type Status struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	// One of: "success", "failure", "working" (for operations not yet completed)
	// TODO: if "working", include an operation identifier so final status can be
	// checked.
	Status string `json:"status,omitempty" yaml:"status,omitempty"`
	// Details about the status. May be an error description or an
	// operation number for later polling.
	Details string `json:"details,omitempty" yaml:"details,omitempty"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int `json:"code,omitempty" yaml:"code,omitempty"`
}

// Values of Status.Status
const (
	StatusSuccess = "success"
	StatusFailure = "failure"
	StatusWorking = "working"
)

// ServerOp is an operation delivered to API clients.
type ServerOp struct {
	api.JSONBase `yaml:",inline" json:",inline"`
}

// ServerOpList is a list of operations, as delivered to API clients.
type ServerOpList struct {
	api.JSONBase `yaml:",inline" json:",inline"`
	Items        []ServerOp `yaml:"items,omitempty" json:"items,omitempty"`
}

// WatchEvent objects are streamed from the api server in response to a watch request.
type WatchEvent struct {
	// The type of the watch event; added, modified, or deleted.
	Type watch.EventType

	// For added or modified objects, this is the new object; for deleted objects,
	// it's the state of the object immediately prior to its deletion.
	Object APIObject
}

// APIObject has appropriate encoding and decoder functions, such that on the wire, it's
// stored as a []byte, but in memory, the contained object is accessable as an interface{}
// via the Get() function. Only objects having a JSONBase may be stored via APIObject.
// The purpose of this is to allow an API object of type known only at runtime to be
// embedded within other API objects.
type APIObject struct {
	Object   interface{}
	Encoding Encoding
}

// AddTypes adds the types in this package to a conversion scheme
func AddTypes(scheme *conversion.Scheme) {
	//TODO: needs to be fixed
	//scheme.MetaInsertionFactory = metaInsertion{}
	scheme.AddKnownTypes("v1beta1",
		Status{},
		ServerOpList{},
		ServerOp{},
	)
	scheme.AddKnownTypes("",
		Status{},
		ServerOpList{},
		ServerOp{},
	)
}
