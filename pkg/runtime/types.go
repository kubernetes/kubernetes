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

package runtime

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Note that the types provided in this file are not versioned and are intended to be
// safe to use from within all versions of every API object.

// JSONBase is shared by all top level objects. The proper way to use it is to inline it in your type,
// like this:
// type MyAwesomeAPIObject struct {
// 	runtime.JSONBase    `yaml:",inline" json:",inline"`
// 	... // other fields
// }
//
// JSONBase is provided here for convenience. You may use it directlly from this package or define
// your own with the same fields.
//
type JSONBase struct {
	Kind              string    `json:"kind,omitempty" yaml:"kind,omitempty"`
	ID                string    `json:"id,omitempty" yaml:"id,omitempty"`
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
	SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
	ResourceVersion   uint64    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
	APIVersion        string    `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
}

// Object has appropriate encoder and decoder functions, such that on the wire, it's
// stored as a []byte, but in memory, the contained object is accessable as an interface{}
// via the Get() function. Only objects having a JSONBase may be stored via Object.
// The purpose of this is to allow an API object of type known only at runtime to be
// embedded within other API objects.
//
// Note that object assumes that you've registered all of your api types with the api package.
//
// Note that objects will be serialized into the api package's default external versioned type;
// this should be fixed in the future to use the version of the current Codec instead.
type Object struct {
	Object interface{}
}

// Extension allows api objects with unknown types to be passed-through. This can be used
// to deal with the API objects from a plug-in. Extension objects still have functioning
// JSONBase features-- kind, version, resourceVersion, etc.
// TODO: Not implemented yet
type Extension struct {
}
