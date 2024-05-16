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
	"errors"
	"path"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
)

// ToSchemaFromOpenAPI converts a directory of OpenAPI schemas to an smd Schema.
//   - models: a map from definition name to OpenAPI V3 structural schema for each definition.
//     Key in map is used to resolve references in the schema.
//   - preserveUnknownFields: flag indicating whether unknown fields in all schemas should be preserved.
//   - returns: nil and an error if there is a parse error, or if schema does not satisfy a
//     required structural schema invariant for conversion. If no error, returns
//     a new smd schema.
//
// Schema should be validated as structural before using with this function, or
// there may be information lost.
func ToSchemaFromOpenAPI(models map[string]*spec.Schema, preserveUnknownFields bool) (*schema.Schema, error) {
	c := convert{
		preserveUnknownFields: preserveUnknownFields,
		output:                &schema.Schema{},
	}

	for name, spec := range models {
		// Skip/Ignore top-level references
		if len(spec.Ref.String()) > 0 {
			continue
		}

		var a schema.Atom

		// Hard-coded schemas for now as proto_models implementation functions.
		// https://github.com/kubernetes/kube-openapi/issues/364
		if name == quantityResource {
			a = schema.Atom{
				Scalar: untypedDef.Atom.Scalar,
			}
		} else if name == rawExtensionResource {
			a = untypedDef.Atom
		} else {
			c2 := c.push(name, &a)
			c2.visitSpec(spec)
			c.pop(c2)
		}

		c.insertTypeDef(name, a)
	}

	if len(c.errorMessages) > 0 {
		return nil, errors.New(strings.Join(c.errorMessages, "\n"))
	}

	c.addCommonTypes()
	return c.output, nil
}

func (c *convert) visitSpec(m *spec.Schema) {
	// Check if this schema opts its descendants into preserve-unknown-fields
	if p, ok := m.Extensions["x-kubernetes-preserve-unknown-fields"]; ok && p == true {
		c.preserveUnknownFields = true
	}
	a := c.top()
	*a = c.parseSchema(m)
}

func (c *convert) parseSchema(m *spec.Schema) schema.Atom {
	// k8s-generated OpenAPI specs have historically used only one value for
	// type and starting with OpenAPIV3 it is only allowed to be
	// a single string.
	typ := ""
	if len(m.Type) > 0 {
		typ = m.Type[0]
	}

	// Structural Schemas produced by kubernetes follow very specific rules which
	// we can use to infer the SMD type:
	switch typ {
	case "":
		// According to Swagger docs:
		// https://swagger.io/docs/specification/data-models/data-types/#any
		//
		// If no type is specified, it is equivalent to accepting any type.
		return schema.Atom{
			Scalar: ptr(schema.Scalar("untyped")),
			List:   c.parseList(m),
			Map:    c.parseObject(m),
		}

	case "object":
		return schema.Atom{
			Map: c.parseObject(m),
		}
	case "array":
		return schema.Atom{
			List: c.parseList(m),
		}
	case "integer", "boolean", "number", "string":
		return convertPrimitive(typ, m.Format)
	default:
		c.reportError("unrecognized type: '%v'", typ)
		return schema.Atom{
			Scalar: ptr(schema.Scalar("untyped")),
		}
	}
}

func (c *convert) makeOpenAPIRef(specSchema *spec.Schema) schema.TypeRef {
	refString := specSchema.Ref.String()

	// Special-case handling for $ref stored inside a single-element allOf
	if len(refString) == 0 && len(specSchema.AllOf) == 1 && len(specSchema.AllOf[0].Ref.String()) > 0 {
		refString = specSchema.AllOf[0].Ref.String()
	}

	if _, n := path.Split(refString); len(n) > 0 {
		//!TODO: Refactor the field ElementRelationship override
		// we can generate the types with overrides ahead of time rather than
		// requiring the hacky runtime support
		// (could just create a normalized key struct containing all customizations
		// 	to deduplicate)
		mapRelationship, err := getMapElementRelationship(specSchema.Extensions)
		if err != nil {
			c.reportError(err.Error())
		}

		if len(mapRelationship) > 0 {
			return schema.TypeRef{
				NamedType:           &n,
				ElementRelationship: &mapRelationship,
			}
		}

		return schema.TypeRef{
			NamedType: &n,
		}

	}
	var inlined schema.Atom

	// compute the type inline
	c2 := c.push("inlined in "+c.currentName, &inlined)
	c2.preserveUnknownFields = c.preserveUnknownFields
	c2.visitSpec(specSchema)
	c.pop(c2)

	return schema.TypeRef{
		Inlined: inlined,
	}
}

func (c *convert) parseObject(s *spec.Schema) *schema.Map {
	var fields []schema.StructField
	for name, member := range s.Properties {
		fields = append(fields, schema.StructField{
			Name:    name,
			Type:    c.makeOpenAPIRef(&member),
			Default: member.Default,
		})
	}

	// AdditionalProperties informs the schema of any "unknown" keys
	// Unknown keys are enforced by the ElementType field.
	elementType := func() schema.TypeRef {
		if s.AdditionalProperties == nil {
			// According to openAPI spec, an object without properties and without
			// additionalProperties is assumed to be a free-form object.
			if c.preserveUnknownFields || len(s.Properties) == 0 {
				return schema.TypeRef{
					NamedType: &deducedName,
				}
			}

			// If properties are specified, do not implicitly allow unknown
			// fields
			return schema.TypeRef{}
		} else if s.AdditionalProperties.Schema != nil {
			// Unknown fields use the referred schema
			return c.makeOpenAPIRef(s.AdditionalProperties.Schema)

		} else if s.AdditionalProperties.Allows {
			// A boolean instead of a schema was provided. Deduce the
			// type from the value provided at runtime.
			return schema.TypeRef{
				NamedType: &deducedName,
			}
		} else {
			// Additional Properties are explicitly disallowed by the user.
			// Ensure element type is empty.
			return schema.TypeRef{}
		}
	}()

	relationship, err := getMapElementRelationship(s.Extensions)
	if err != nil {
		c.reportError(err.Error())
	}

	return &schema.Map{
		Fields:              fields,
		ElementRelationship: relationship,
		ElementType:         elementType,
	}
}

func (c *convert) parseList(s *spec.Schema) *schema.List {
	relationship, mapKeys, err := getListElementRelationship(s.Extensions)
	if err != nil {
		c.reportError(err.Error())
	}
	elementType := func() schema.TypeRef {
		if s.Items != nil {
			if s.Items.Schema == nil || s.Items.Len() != 1 {
				c.reportError("structural schema arrays must have exactly one member subtype")
				return schema.TypeRef{
					NamedType: &deducedName,
				}
			}

			subSchema := s.Items.Schema
			if subSchema == nil {
				subSchema = &s.Items.Schemas[0]
			}
			return c.makeOpenAPIRef(subSchema)
		} else if len(s.Type) > 0 && len(s.Type[0]) > 0 {
			c.reportError("`items` must be specified on arrays")
		}

		// A list with no items specified is treated as "untyped".
		return schema.TypeRef{
			NamedType: &untypedName,
		}

	}()

	return &schema.List{
		ElementRelationship: relationship,
		Keys:                mapKeys,
		ElementType:         elementType,
	}
}
