/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"k8s.io/kubernetes/pkg/runtime"
)

// ObjectFunc is a function to act on a given object. An error may be returned
// if the hook cannot be completed. An ObjectFunc may transform the provided
// object.
type ObjectFunc func(obj runtime.Object) error

// AllFuncs returns an ObjectFunc that attempts to run all of the provided functions
// in order, returning early if there are any errors.
func AllFuncs(fns ...ObjectFunc) ObjectFunc {
	return func(obj runtime.Object) error {
		for _, fn := range fns {
			if fn == nil {
				continue
			}
			if err := fn(obj); err != nil {
				return err
			}
		}
		return nil
	}
}
