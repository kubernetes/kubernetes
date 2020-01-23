/*
Copyright 2019 The Kubernetes Authors.

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

package value

import "reflect"

type listReflect struct {
	Value reflect.Value
}

func (r listReflect) Length() int {
	val := r.Value
	return val.Len()
}

func (r listReflect) At(i int) Value {
	val := r.Value
	return mustWrapValueReflect(val.Index(i))
}

func (r listReflect) Unstructured() interface{} {
	l := r.Length()
	result := make([]interface{}, l)
	for i := 0; i < l; i++ {
		result[i] = r.At(i).Unstructured()
	}
	return result
}
