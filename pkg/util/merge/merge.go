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

package merge

import (
	"encoding/json"
)

// MergeJSON merges JSON according to RFC7386
// (see https://tools.ietf.org/html/rfc7386)
func MergeJSON(dst, src []byte) ([]byte, error) {
	var target interface{}
	if err := json.Unmarshal(dst, &target); err != nil {
		return nil, err
	}
	var patch interface{}
	if err := json.Unmarshal(src, &patch); err != nil {
		return nil, err
	}
	return json.Marshal(MergePatch(target, patch))
}

// MergePatch is an implementation of MergePatch described in RFC7386 that operates on
// json marshalled into empty interface{} by encoding/json.Unmarshal()
// (see https://tools.ietf.org/html/rfc7386#section-2)
func MergePatch(target, patch interface{}) interface{} {
	if patchObject, isPatchObject := patch.(map[string]interface{}); isPatchObject {
		targetObject := make(map[string]interface{})
		if m, isTargetObject := target.(map[string]interface{}); isTargetObject {
			targetObject = m
		}
		for name, value := range patchObject {
			if _, found := targetObject[name]; value == nil && found {
				delete(targetObject, name)
			} else {
				targetObject[name] = MergePatch(targetObject[name], value)
			}
		}
		return targetObject
	} else {
		return patch
	}
}
