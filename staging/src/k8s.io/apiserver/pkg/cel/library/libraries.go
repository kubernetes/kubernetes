/*
Copyright 2024 The Kubernetes Authors.

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

package library

import (
	"github.com/google/cel-go/cel"
)

// Library represents a CEL library used by kubernetes.
type Library interface {
	// SingletonLibrary provides the library name and ensures the library can be safely registered into environments.
	cel.SingletonLibrary

	// Types provides all custom types introduced by the library.
	Types() []*cel.Type

	// declarations returns all function declarations provided by the library.
	declarations() map[string][]cel.FunctionOpt
}

// KnownLibraries returns all libraries used in Kubernetes.
func KnownLibraries() []Library {
	return []Library{
		authzLib,
		authzSelectorsLib,
		listsLib,
		regexLib,
		urlsLib,
		quantityLib,
		ipLib,
		cidrsLib,
		formatLib,
		semverLib,
		jsonPatchLib,
	}
}

func isRegisteredType(typeName string) bool {
	for _, lib := range KnownLibraries() {
		for _, rt := range lib.Types() {
			if rt.TypeName() == typeName {
				return true
			}
		}
	}
	return false
}
