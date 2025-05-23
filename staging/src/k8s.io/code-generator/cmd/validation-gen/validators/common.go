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

package validators

import (
	"k8s.io/gengo/v2/types"
)

const (
	// libValidationPkg is the pkgpath to our "standard library" of validation
	// functions.
	libValidationPkg = "k8s.io/apimachinery/pkg/api/validate"
)

// rootTypeString returns a string representation of the relationship between
// src and dst types, for use in error messages.
func rootTypeString(src, dst *types.Type) string {
	if src == dst {
		return src.String()
	}
	return src.String() + " -> " + dst.String()
}

// IsDirectComparable returns true if the type is safe to compare using "==".
// It is similar to gengo.IsComparable, but it doesn't consider Pointers to be
// comparable (we don't want shallow compare).
func IsDirectComparable(t *types.Type) bool {
	switch t.Kind {
	case types.Builtin:
		return true
	case types.Struct:
		for _, f := range t.Members {
			if !IsDirectComparable(f.Type) {
				return false
			}
		}
		return true
	case types.Array:
		return IsDirectComparable(t.Elem)
	case types.Alias:
		return IsDirectComparable(t.Underlying)
	}
	return false
}
