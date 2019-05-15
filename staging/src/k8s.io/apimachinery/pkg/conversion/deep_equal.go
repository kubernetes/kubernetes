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
	"reflect"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	kubereflect "k8s.io/apimachinery/third_party/forked/golang/reflect"
)

// The code for this type must be located in third_party, since it forks from
// go std lib. But for convenience, we expose the type here, too.
type Equalities struct {
	eq   kubereflect.Equalities
	opts cmp.Options
}

// For convenience, panics on errors
func EqualitiesOrDie(funcs ...interface{}) Equalities {
	e := Equalities{eq: kubereflect.EqualitiesOrDie(funcs...)}

	for _, f := range funcs {
		e.opts = append(e.opts, cmp.Comparer(f))
	}
	// An empty slice *is* equal to a nil slice for our purposes; same for maps.
	e.opts = append(e.opts, cmpopts.EquateEmpty())

	return e
}

// DeepEqual is like reflect.DeepEqual, but focused on semantic equality
// instead of memory equality.
//
// It will use e's equality functions if it finds types that match.
//
// An empty slice *is* equal to a nil slice for our purposes; same for maps.
//
// Unexported field members cannot be compared and will cause an imformative
// panic; you must add an Equality function for these types.
func (e Equalities) DeepEqual(a, b interface{}) bool {
	return e.eq.DeepEqual(a, b)
}

// DeepDerivative is similar to DeepEqual except that unset fields in "a" are
// ignored (not compared). This allows us to focus on the fields that matter to
// the semantic comparison.
//
// The unset fields include a nil pointer and an empty string.
func (e Equalities) DeepDerivative(a, b interface{}) bool {
	return cmp.Equal(a, b, e.opts, ignoreUnset)
}

var ignoreUnset = cmp.Options{
	// ignore unset fields in v1
	cmp.FilterPath(func(path cmp.Path) bool {
		v1, _ := path.Last().Values()
		switch v1.Kind() {
		case reflect.Slice, reflect.Map:
			if v1.IsNil() || v1.Len() == 0 {
				return true
			}
		case reflect.String:
			if v1.Len() == 0 {
				return true
			}
		case reflect.Interface, reflect.Ptr:
			if v1.IsNil() {
				return true
			}
		}
		return false
	}, cmp.Ignore()),
	// ignore map entries that aren't set in v1
	cmp.FilterPath(func(path cmp.Path) bool {
		switch i := path.Last().(type) {
		case cmp.MapIndex:
			if v1, _ := i.Values(); !v1.IsValid() {
				return true
			}
		}
		return false
	}, cmp.Ignore()),
}
