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
	"gopkg.in/v1/yaml"
)

// Encode()/Decode() are the canonical way of converting an API object to/from
// wire format. This file provides utility functions which permit doing so
// recursively, such that API objects of types known only at run time can be
// embedded within other API types.

// UnmarshalJSON implements the json.Unmarshaler interface.
func (a *APIObject) UnmarshalJSON(b []byte) error {
	// Handle JSON's "null": Decode() doesn't expect it.
	if len(b) == 4 && string(b) == "null" {
		a.Object = nil
		return nil
	}

	obj, err := Decode(b)
	if err != nil {
		return err
	}
	a.Object = obj
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (a APIObject) MarshalJSON() ([]byte, error) {
	if a.Object == nil {
		// Encode unset/nil objects as JSON's "null".
		return []byte("null"), nil
	}

	return Encode(a.Object)
}

// SetYAML implements the yaml.Setter interface.
func (a *APIObject) SetYAML(tag string, value interface{}) bool {
	if value == nil {
		a.Object = nil
		return true
	}
	// Why does the yaml package send value as a map[interface{}]interface{}?
	// It's especially frustrating because encoding/json does the right thing
	// by giving a []byte. So here we do the embarrasing thing of re-encode and
	// de-encode the right way.
	// TODO: Write a version of Decode that uses reflect to turn this value
	// into an API object.
	b, err := yaml.Marshal(value)
	if err != nil {
		panic("yaml can't reverse its own object")
	}
	obj, err := Decode(b)
	if err != nil {
		return false
	}
	a.Object = obj
	return true
}

// GetYAML implements the yaml.Getter interface.
func (a APIObject) GetYAML() (tag string, value interface{}) {
	if a.Object == nil {
		value = "null"
		return
	}
	// Encode returns JSON, which is conveniently a subset of YAML.
	v, err := Encode(a.Object)
	if err != nil {
		panic("impossible to encode API object!")
	}
	return tag, v
}
