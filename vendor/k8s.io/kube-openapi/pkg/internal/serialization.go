/*
Copyright 2023 The Kubernetes Authors.

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

package internal

import (
	"github.com/go-openapi/jsonreference"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// DeterministicMarshal calls the jsonv2 library with the deterministic
// flag in order to have stable marshaling.
func DeterministicMarshal(in any) ([]byte, error) {
	return jsonv2.MarshalOptions{Deterministic: true}.Marshal(jsonv2.EncodeOptions{}, in)
}

// JSONRefFromMap populates a json reference object if the map v contains a $ref key.
func JSONRefFromMap(jsonRef *jsonreference.Ref, v map[string]interface{}) error {
	if v == nil {
		return nil
	}
	if vv, ok := v["$ref"]; ok {
		if str, ok := vv.(string); ok {
			ref, err := jsonreference.New(str)
			if err != nil {
				return err
			}
			*jsonRef = ref
		}
	}
	return nil
}

// SanitizeExtensions sanitizes the input map such that non extension
// keys (non x-*, X-*) keys are dropped from the map. Returns the new
// modified map, or nil if the map is now empty.
func SanitizeExtensions(e map[string]interface{}) map[string]interface{} {
	for k := range e {
		if !IsExtensionKey(k) {
			delete(e, k)
		}
	}
	if len(e) == 0 {
		e = nil
	}
	return e
}

// IsExtensionKey returns true if the input string is of format x-* or X-*
func IsExtensionKey(k string) bool {
	return len(k) > 1 && (k[0] == 'x' || k[0] == 'X') && k[1] == '-'
}
