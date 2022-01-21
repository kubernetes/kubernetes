// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"github.com/google/cel-go/common/types/ref"
)

// IsUnknownOrError returns whether the input element ref.Val is an ErrType or UnknonwType.
func IsUnknownOrError(val ref.Val) bool {
	switch val.Type() {
	case UnknownType, ErrType:
		return true
	}
	return false
}

// IsPrimitiveType returns whether the input element ref.Val is a primitive type.
// Note, primitive types do not include well-known types such as Duration and Timestamp.
func IsPrimitiveType(val ref.Val) bool {
	switch val.Type() {
	case BoolType, BytesType, DoubleType, IntType, StringType, UintType:
		return true
	}
	return false
}
