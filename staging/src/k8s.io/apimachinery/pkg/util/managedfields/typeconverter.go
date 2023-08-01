/*
Copyright 2018 The Kubernetes Authors.

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

package managedfields

import (
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// TypeConverter allows you to convert from runtime.Object to
// typed.TypedValue and the other way around.
type TypeConverter = internal.TypeConverter

// NewDeducedTypeConverter creates a TypeConverter for CRDs that don't
// have a schema. It does implement the same interface though (and
// create the same types of objects), so that everything can still work
// the same. CRDs are merged with all their fields being "atomic" (lists
// included).
func NewDeducedTypeConverter() TypeConverter {
	return internal.NewDeducedTypeConverter()
}

// NewTypeConverter builds a TypeConverter from a map of OpenAPIV3 schemas.
// This will automatically find the proper version of the object, and the
// corresponding schema information.
// The keys to the map must be consistent with the names
// used by Refs within the schemas.
// The schemas should conform to the Kubernetes Structural Schema OpenAPI
// restrictions found in docs:
// https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#specifying-a-structural-schema
func NewTypeConverter(openapiSpec map[string]*spec.Schema, preserveUnknownFields bool) (TypeConverter, error) {
	return internal.NewTypeConverter(openapiSpec, preserveUnknownFields)
}
