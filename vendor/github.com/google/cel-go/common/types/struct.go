// Copyright 2026 Google LLC
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
	"reflect"

	"github.com/google/cel-go/common/types/ref"
)

// StructTypeDescriptor describes a CEL struct type, providing field metadata and value instantiation.
type StructTypeDescriptor interface {
	// ReflectType returns the backing Go reflect.Type associated with the struct (or nil if non-reflected).
	ReflectType() reflect.Type

	// FieldNames returns the list of field names defined on the struct.
	FieldNames() []string

	// FindFieldType returns the field type and a boolean indicating if the field exists.
	FindFieldType(fieldName string) (*FieldType, bool)

	// NewValue creates a new CEL struct value from the given map of field values.
	NewValue(adapter Adapter, fields map[string]ref.Val) ref.Val

	// Adapt converts a native Go value (struct instance or pointer) to a CEL ref.Val.
	Adapt(adapter Adapter, value any) ref.Val
}
