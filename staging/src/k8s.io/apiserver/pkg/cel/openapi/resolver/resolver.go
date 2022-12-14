/*
Copyright 2023 The Kubernetes Authors.

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

package resolver

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// SchemaResolver finds the OpenAPI schema for the given GroupVersionKind.
// This interface uses the type defined by k8s.io/kube-openapi
type SchemaResolver interface {
	// ResolveSchema takes a GroupVersionKind (GVK) and returns the OpenAPI schema
	// identified by the GVK.
	// The function returns a non-nil error if the schema cannot be found or fail
	// to resolve. The returned error wraps ErrSchemaNotFound if the resolution is
	// attempted but the corresponding schema cannot be found.
	ResolveSchema(gvk schema.GroupVersionKind) (*spec.Schema, error)
}

// ErrSchemaNotFound is wrapped and returned if the schema cannot be located
// by the resolver.
var ErrSchemaNotFound = fmt.Errorf("schema not found")
