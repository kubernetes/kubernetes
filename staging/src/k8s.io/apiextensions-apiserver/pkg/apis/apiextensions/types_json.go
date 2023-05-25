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

package apiextensions

import "k8s.io/apimachinery/pkg/runtime"

// JSON represents any valid JSON value.
// These types are supported: bool, int64, float64, string, []interface{}, map[string]interface{} and nil.
type JSON struct {
	Object interface{}
}

func (j *JSON) DeepCopy() *JSON {
	if j == nil {
		return nil
	}
	return &JSON{Object: runtime.DeepCopyJSONValue(j.Object)}
}

func (j *JSON) DeepCopyInto(target *JSON) {
	if target == nil {
		return
	}
	if j == nil {
		target.Object = nil // shouldn't happen
	}
	target.Object = runtime.DeepCopyJSONValue(j.Object)
}
