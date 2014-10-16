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
	"errors"

	"gopkg.in/v1/yaml"
)

func (re *RawExtension) UnmarshalJSON(in []byte) error {
	if re == nil {
		return errors.New("runtime.RawExtension: UnmarshalJSON on nil pointer")
	}
	re.RawJSON = append(re.RawJSON[0:0], in...)
	return nil
}

func (re *RawExtension) MarshalJSON() ([]byte, error) {
	return re.RawJSON, nil
}

// SetYAML implements the yaml.Setter interface.
func (re *RawExtension) SetYAML(tag string, value interface{}) bool {
	if value == nil {
		re.RawJSON = []byte("null")
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
	re.RawJSON = b
	return true
}

// GetYAML implements the yaml.Getter interface.
func (re *RawExtension) GetYAML() (tag string, value interface{}) {
	return tag, re.RawJSON
}
