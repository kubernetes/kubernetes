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
	"gopkg.in/v1/yaml"
)

// Encode()/Decode() are the canonical way of converting an API object to/from
// wire format. This file provides utility functions which permit doing so
// recursively, such that API objects of types known only at run time can be
// embedded within other API types.

// UnmarshalJSON implements the json.Unmarshaler interface.
func CodecUnmarshalJSON(codec Codec, b []byte) (Object, error) {
	// Handle JSON's "null": Decode() doesn't expect it.
	if len(b) == 4 && string(b) == "null" {
		return nil, nil
	}

	obj, err := codec.Decode(b)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

// MarshalJSON implements the json.Marshaler interface.
func CodecMarshalJSON(codec Codec, obj Object) ([]byte, error) {
	if obj == nil {
		// Encode unset/nil objects as JSON's "null".
		return []byte("null"), nil
	}

	return codec.Encode(obj)
}

// SetYAML implements the yaml.Setter interface.
func CodecSetYAML(codec Codec, tag string, value interface{}) (Object, bool) {
	if value == nil {
		return nil, true
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
	obj, err := codec.Decode(b)
	if err != nil {
		return nil, false
	}
	return obj, true
}

// GetYAML implements the yaml.Getter interface.
func CodecGetYAML(codec Codec, obj Object) (tag string, value interface{}) {
	if obj == nil {
		value = "null"
		return
	}
	// Encode returns JSON, which is conveniently a subset of YAML.
	v, err := codec.Encode(obj)
	if err != nil {
		panic("impossible to encode API object!")
	}
	return tag, v
}
