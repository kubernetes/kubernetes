/*
Copyright The Kubernetes Authors.

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

import "encoding/json"

// MarshalJSON() preserves pre-1.34 JSON encoding of value as a string (not base64),
// stomping non-utf-8 data with the utf8 replacement character.
func (k *KeyValue) MarshalJSON() ([]byte, error) {
	return json.Marshal(stringKeyValue{
		Key:   k.GetKey(),
		Value: string(k.GetValue()),
	})
}

// UnmarshalJSON preserves pre-1.34 JSON decoding of value as a string (not base64),
// stomping non-utf-8 data with the utf8 replacement character.
func (k *KeyValue) UnmarshalJSON(data []byte) error {
	v := stringKeyValue{}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	k.Key = v.Key
	k.Value = []byte(v.Value)
	return nil
}

// stringKeyValue matches the structure used to json-encode pre-1.34.
// Non-UTF-8 characters in Value are coerced to the replacement character on encode/decode.
type stringKeyValue struct {
	Key   string `json:"key,omitempty"`
	Value string `json:"value,omitempty"`
}
