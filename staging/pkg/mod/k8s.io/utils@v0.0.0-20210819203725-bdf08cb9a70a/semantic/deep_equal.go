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

package semantic

import (
	"k8s.io/utils/third_party/forked/golang/reflect"
)

// Equalities is a map from type to a function comparing two values of
// that type.
type Equalities = reflect.Equalities

// EqualitiesOrDie adds the given funcs and panics on any error.
func EqualitiesOrDie(funcs ...interface{}) Equalities {
	return reflect.EqualitiesOrDie(funcs...)
}
