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

package unstructured

import (
	"fmt"

	"github.com/google/cel-go/common/types"
)

// NewFieldType creates a field by its field name.
// This version of FieldType is unstructured and has DynType as its type.
func NewFieldType(name string) *types.FieldType {
	return &types.FieldType{
		// for unstructured, we do not check for its type,
		// use DynType for all fields.
		Type: types.DynType,
		IsSet: func(target any) bool {
			// for an unstructured object, we allow any field to be considered set.
			return true
		},
		GetFrom: func(target any) (any, error) {
			if m, ok := target.(map[string]any); ok {
				return m[name], nil
			}
			return nil, fmt.Errorf("cannot get field %q", name)
		},
	}
}
