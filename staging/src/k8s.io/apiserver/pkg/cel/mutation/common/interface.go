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

package common

import (
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// TypeResolver resolves a type by a given name.
type TypeResolver interface {
	// Resolve resolves the type by its name, starting with "Object" as its root.
	// The type that the name refers to must be an object.
	// This function returns false if the name does not refer to a known object type.
	Resolve(name string) (TypeRef, bool)
}

// TypeRef refers an object type that can be looked up for its fields.
type TypeRef interface {
	ref.Type

	// CELType wraps the TypeRef to be a type that is understood by CEL.
	CELType() *types.Type

	// Field finds the field by the field name, or false if the field is not known.
	// This function directly return a FieldType that is known to CEL to be more customizable.
	Field(name string) (*types.FieldType, bool)

	// Val creates an instance for the TypeRef, given its fields and their values.
	Val(fields map[string]ref.Val) ref.Val
}
