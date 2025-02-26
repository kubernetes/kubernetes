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

package validate

import (
	"context"
	"slices"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Enum verifies that the specified value is one of the valid symbols.
// This is for string enums only.
func Enum[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T, symbols sets.Set[T]) field.ErrorList {
	if value == nil {
		return nil
	}
	if !symbols.Has(*value) {
		symbolList := symbols.UnsortedList()
		slices.Sort(symbolList)
		return field.ErrorList{field.NotSupported[T](fldPath, *value, symbolList)}
	}
	return nil
}
