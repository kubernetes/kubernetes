/*
Copyright 2017 The Kubernetes Authors.

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

package panic

import utilruntime "k8s.io/apimachinery/pkg/util/runtime"

// HandlePanic returns a function that wraps `fn` with the utilruntime.PanicHandlers, and continues
// to bubble the panic after the PanicHandlers are called
func HandlePanic(fn func()) func() {
	return func() {
		defer func() {
			if r := recover(); r != nil {
				for _, fn := range utilruntime.PanicHandlers {
					fn(r)
				}
				panic(r)
			}
		}()
		// call the function
		fn()
	}
}
