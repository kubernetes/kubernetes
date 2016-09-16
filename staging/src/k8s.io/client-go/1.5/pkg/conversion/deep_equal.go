/*
Copyright 2015 The Kubernetes Authors.

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

package conversion

import (
	"k8s.io/client-go/1.5/pkg/third_party/forked/golang/reflect"
)

// The code for this type must be located in third_party, since it forks from
// go std lib. But for convenience, we expose the type here, too.
type Equalities struct {
	reflect.Equalities
}

// For convenience, panics on errors
func EqualitiesOrDie(funcs ...interface{}) Equalities {
	e := Equalities{reflect.Equalities{}}
	if err := e.AddFuncs(funcs...); err != nil {
		panic(err)
	}
	return e
}
