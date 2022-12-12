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

package schemaconv

import (
	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
)

func (c *convert) VisitSpec(m *spec.Schema) {
	// k8s-generated OpenAPI specs have historically used only one value for
	// type and starting with OpenAPIV3 it is only allowed to be
	// a single string.
	typ := ""
	if len(m.Type) > 0 {
		typ = m.Type[0]
	}

	a := c.top()

	preserveUnknownFields := c.preserveUnknownFields
	if p, ok := m.Extensions["x-kubernetes-preserve-unknown-fields"]; ok && p == true {
		preserveUnknownFields = true
	}

	// Structural Schemas produced by kubernetes follow very specific rules which
	// we can use to infer the SMD type:
	switch typ {
	case "":
		// Some older k8s specs did not include a type for objects. Keep behavior
		// consistent with old proto.Models implementation:
		//	If there are Properties, it is a top-level Kind.
		// 	Otherwise, it is Arbitrary/deduced
		if len(m.Properties) == 0 {
			// Treat same way as Proto.Models:
			// If there are no properties, then treat as arbitrary (deduced)
			*a = deducedDef.Atom
		} else {
			a.Map = &schema.Map{
				ElementType: schema.TypeRef{},
			}

			//TODO: Remove this in a future PR which removes old/unused union
			// code. Kept here for parity with proto models conversion
			unions, err := makeUnions(m.Extensions)
			if err != nil {
				c.reportError(err.Error())
				return
			}
			a.Map.Unions = unions

			if preserveUnknownFields {
				a.Map.ElementType = schema.TypeRef{
					NamedType: &deducedName,
				}
			}

			for name, member := range m.Properties {
				tr := c.makeOpenAPIRef(&member, preserveUnknownFields)
				a.Map.Fields = append(a.Map.Fields, schema.StructField{
					Name:    name,
					Type:    tr,
					Default: member.Default,
				})
			}

			if mapRelationship, err := getMapElementRelationship(m.Extensions); err != nil {
				c.reportError(err.Error())
			} else if len(mapRelationship) > 0 {
				a.Map.ElementRelationship = mapRelationship
			}
		}
	case "object":
		a.Map = &schema.Map{
			ElementType: schema.TypeRef{},
		}

		// Gate against non-empty properties to mirror protoModels behavior
		// until this is removed
		if len(m.Properties) > 0 {
			//TODO: Remove this in a future PR which removes old/unused union
			// code. Kept here for parity with proto models conversion
			unions, err := makeUnions(m.Extensions)
			if err != nil {
				c.reportError(err.Error())
				return
			}
			a.Map.Unions = unions
		}

		// Slight difference from old proto.Models behavior:
		// here we allow Properties and AdditionalProperties to be defined
		// together.
		//
		// This was an incompleteness in old implementation.
		for name, member := range m.Properties {
			tr := c.makeOpenAPIRef(&member, preserveUnknownFields)
			a.Map.Fields = append(a.Map.Fields, schema.StructField{
				Name:    name,
				Type:    tr,
				Default: member.Default,
			})
		}

		// AdditionalProperties informs the schema of any "unknown" keys
		// Unknown keys are enforced by the ElementType field.
		if m.AdditionalProperties == nil {
			// This mirrors how older spec -> proto.Models -> smd would behave.
			//
			// proto.Models would unconditionally treat any "object" with empty
			// "properties"&"additionalProperties" as a Map allowing unknown
			// keys...I dont think this was explicitly supported behavior.
			// In Structural Schema docs I've seen no distinction between object
			// schemas that define properties and those that do not.
			//
			// One example of first party schema making use of
			// this is "io.k8s.apimachinery.pkg.apis.meta.v1.FieldsV1"
			//
			// If additional properties is not provided, I would expect
			// unknown keys to be not allowed unless preserveUnknownFields.
			if preserveUnknownFields || len(m.Properties) == 0 {
				a.Map.ElementType = schema.TypeRef{
					NamedType: &deducedName,
				}
			} else {
				// If we are not to preserve unknown fields, then make sure
				// are not allowed.
				a.Map.ElementType = schema.TypeRef{}
			}
		} else if m.AdditionalProperties.Schema != nil {
			// Unknown fields use the referred schema
			a.Map.ElementType = c.makeOpenAPIRef(m.AdditionalProperties.Schema, preserveUnknownFields)

		} else if m.AdditionalProperties.Allows {
			// A boolean instead of a schema was provided. Deduce the
			// type from the value provided at runtime.
			//
			// This differs from old proto.Models implementation which
			// left this as incomplete
			a.Map.ElementType = schema.TypeRef{
				NamedType: &deducedName,
			}
		} else {
			// Additional Properties are explicitly disallowed by the user.
			// Ensure element type is empty.
			a.Map.ElementType = schema.TypeRef{}
		}

		if mapRelationship, err := getMapElementRelationship(m.Extensions); err != nil {
			c.reportError(err.Error())
		} else if len(mapRelationship) > 0 {
			a.Map.ElementRelationship = mapRelationship
		}

	case "array":
		if m.Items.Schema == nil && m.Items.Len() != 1 {
			c.reportError("structural schema arrays must have exactly one member subtype")
			return
		}

		subSchema := m.Items.Schema
		if subSchema == nil {
			subSchema = &m.Items.Schemas[0]
		}

		// Slight semantic different from proto.Models: preserveUnknownFields
		// extension now being respected for array
		converted, err := convertArray(m.Extensions, c.makeOpenAPIRef(subSchema, preserveUnknownFields))

		if err != nil {
			c.reportError(err.Error())
			a.Scalar = ptr(schema.Scalar("untyped"))
		} else {
			*a = converted
		}

	case "integer", "boolean", "number", "string":
		if c.currentName == quantityResource {
			//!TODO: is this not possible to encode via marker comments instead of
			// in source? This mirrors old proto.Models behavior
			// Why not a new openapi format like we used for int-or-string?
			a.Scalar = ptr(schema.Scalar("untyped"))
		} else {
			*a = convertPrimitive(typ, m.Format)
		}

	default:
		c.reportError("unrecognized type: '%v'", typ)
		a.Scalar = ptr(schema.Scalar("untyped"))
	}
}
