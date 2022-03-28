/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"reflect"

	"k8s.io/kube-openapi/pkg/schemamutation"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// wrapRefs wraps OpenAPI V3 Schema refs that contain sibling elements.
// AllOf is used to wrap the Ref to prevent references from having sibling elements
// Please see https://github.com/kubernetes/kubernetes/issues/106387#issuecomment-967640388
func WrapRefs(schema *spec.Schema) *spec.Schema {
	walker := schemamutation.Walker{
		SchemaCallback: func(schema *spec.Schema) *spec.Schema {
			orig := schema
			clone := func() {
				if orig == schema {
					schema = new(spec.Schema)
					*schema = *orig
				}
			}
			if schema.Ref.String() != "" && !reflect.DeepEqual(*schema, spec.Schema{SchemaProps: spec.SchemaProps{Ref: schema.Ref}}) {
				clone()
				refSchema := new(spec.Schema)
				refSchema.Ref = schema.Ref
				schema.Ref = spec.Ref{}
				schema.AllOf = []spec.Schema{*refSchema}
			}
			return schema
		},
		RefCallback: schemamutation.RefCallbackNoop,
	}
	return walker.WalkSchema(schema)
}
