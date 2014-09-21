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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Codec is the identity codec for this package - it can only convert itself
// to itself.
var Codec = runtime.CodecFor(Scheme, "")

// EmbeddedObject implements a Codec specific version of an
// embedded object.
type EmbeddedObject struct {
	runtime.Object
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (a *EmbeddedObject) UnmarshalJSON(b []byte) error {
	obj, err := runtime.CodecUnmarshalJSON(Codec, b)
	a.Object = obj
	return err
}

// MarshalJSON implements the json.Marshaler interface.
func (a EmbeddedObject) MarshalJSON() ([]byte, error) {
	return runtime.CodecMarshalJSON(Codec, a.Object)
}

// SetYAML implements the yaml.Setter interface.
func (a *EmbeddedObject) SetYAML(tag string, value interface{}) bool {
	obj, ok := runtime.CodecSetYAML(Codec, tag, value)
	a.Object = obj
	return ok
}

// GetYAML implements the yaml.Getter interface.
func (a EmbeddedObject) GetYAML() (tag string, value interface{}) {
	return runtime.CodecGetYAML(Codec, a.Object)
}
