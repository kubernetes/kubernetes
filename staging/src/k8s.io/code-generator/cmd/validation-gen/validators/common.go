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
	"fmt"
	"strconv"

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

// isUnsignedInt returns true if t is an unsigned integer type.
func isUnsignedInt(t *types.Type) bool {
	switch t {
	case types.Uint, types.Uint64, types.Uint32, types.Uint16, types.Byte:
		return true
	}
	return false
}

// intBitSize returns the bit width of a gengo integer type. For platform-sized
// types (int, uint), strconv.IntSize is used. Byte maps to 8. Returns an error
// for any type not in the supported integer set so unrecognized types are
// caught at codegen time rather than silently falling back.
func intBitSize(t *types.Type) (int, error) {
	switch t {
	case types.Byte: // int8 becomes byte in gengo
		return 8, nil
	case types.Int16, types.Uint16:
		return 16, nil
	case types.Int32, types.Uint32:
		return 32, nil
	case types.Int64, types.Uint64:
		return 64, nil
	case types.Int, types.Uint:
		return strconv.IntSize, nil
	default:
		return 0, fmt.Errorf("unsupported integer type: %v", t)
	}
}
