/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"path"
	"sort"
	"strings"

	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
)

const (
	quantityResource = "io.k8s.apimachinery.pkg.api.resource.Quantity"
)

// ToSchema converts openapi definitions into a schema suitable for structured
// merge (i.e. kubectl apply v2).
func ToSchema(models proto.Models) (*schema.Schema, error) {
	return ToSchemaWithPreserveUnknownFields(models, false)
}

// ToSchemaWithPreserveUnknownFields converts openapi definitions into a schema suitable for structured
// merge (i.e. kubectl apply v2), it will preserve unknown fields if specified.
func ToSchemaWithPreserveUnknownFields(models proto.Models, preserveUnknownFields bool) (*schema.Schema, error) {
	c := convert{
		preserveUnknownFields: preserveUnknownFields,
		output:                &schema.Schema{},
	}
	for _, name := range models.ListModels() {
		model := models.LookupModel(name)

		var a schema.Atom
		c2 := c.push(name, &a)
		model.Accept(c2)
		c.pop(c2)

		c.insertTypeDef(name, a)
	}

	if len(c.errorMessages) > 0 {
		return nil, errors.New(strings.Join(c.errorMessages, "\n"))
	}

	c.addCommonTypes()
	return c.output, nil
}

func OpenAPIv3ToSchema(models map[string]spec.Schema, preserveUnknownFields bool) (*schema.Schema, error) {
	c := convert{
		preserveUnknownFields: preserveUnknownFields,
		output:                &schema.Schema{},
	}

	for name, spec := range models {
		// Skip top-level references (mirror behavior of original converter)
		if len(spec.Ref.String()) > 0 {
			continue
		}

		var a schema.Atom
		c2 := c.push(name, &a)
		c2.VisitSpec(&spec)
		c.pop(c2)

		c.insertTypeDef(name, a)
	}

	if len(c.errorMessages) > 0 {
		return nil, errors.New(strings.Join(c.errorMessages, "\n"))
	}

	c.addCommonTypes()
	return c.output, nil
}

type convert struct {
	preserveUnknownFields bool
	output                *schema.Schema

	currentName   string
	current       *schema.Atom
	errorMessages []string
}

func (c *convert) push(name string, a *schema.Atom) *convert {
	return &convert{
		preserveUnknownFields: c.preserveUnknownFields,
		output:                c.output,
		currentName:           name,
		current:               a,
	}
}

func (c *convert) top() *schema.Atom { return c.current }

func (c *convert) pop(c2 *convert) {
	c.errorMessages = append(c.errorMessages, c2.errorMessages...)
}

func (c *convert) reportError(format string, args ...interface{}) {
	c.errorMessages = append(c.errorMessages,
		c.currentName+": "+fmt.Sprintf(format, args...),
	)
}

func (c *convert) insertTypeDef(name string, atom schema.Atom) {
	def := schema.TypeDef{
		Name: name,
		Atom: atom,
	}
	if def.Atom == (schema.Atom{}) {
		// This could happen if there were a top-level reference.
		return
	}
	c.output.Types = append(c.output.Types, def)
}

func (c *convert) addCommonTypes() {
	c.output.Types = append(c.output.Types, untypedDef)
	c.output.Types = append(c.output.Types, deducedDef)
}

var untypedName string = "__untyped_atomic_"

var untypedDef schema.TypeDef = schema.TypeDef{
	Name: untypedName,
	Atom: schema.Atom{
		Scalar: ptr(schema.Scalar("untyped")),
		List: &schema.List{
			ElementType: schema.TypeRef{
				NamedType: &untypedName,
			},
			ElementRelationship: schema.Atomic,
		},
		Map: &schema.Map{
			ElementType: schema.TypeRef{
				NamedType: &untypedName,
			},
			ElementRelationship: schema.Atomic,
		},
	},
}

var deducedName string = "__untyped_deduced_"

var deducedDef schema.TypeDef = schema.TypeDef{
	Name: deducedName,
	Atom: schema.Atom{
		Scalar: ptr(schema.Scalar("untyped")),
		List: &schema.List{
			ElementType: schema.TypeRef{
				NamedType: &untypedName,
			},
			ElementRelationship: schema.Atomic,
		},
		Map: &schema.Map{
			ElementType: schema.TypeRef{
				NamedType: &deducedName,
			},
			ElementRelationship: schema.Separable,
		},
	},
}

func (c *convert) makeRef(model proto.Schema, preserveUnknownFields bool) schema.TypeRef {
	return c.makeAnyRef(model, preserveUnknownFields)
}

func (c *convert) makeOpenAPIRef(model *spec.Schema, preserveUnknownFields bool) schema.TypeRef {
	return c.makeAnyRef(model, preserveUnknownFields)
}

func (c *convert) makeAnyRef(model any, preserveUnknownFields bool) (tr schema.TypeRef) {
	var ext map[string]any
	var refString string

	if specSchema, ok := model.(*spec.Schema); ok {
		ext = specSchema.Extensions
		refString = specSchema.Ref.String()

		// Mirror The protoModels Reference behavior which trims the prefix
		// from the string
		const v2Prefix = "#/definitions/"
		if strings.HasPrefix(refString, v2Prefix) {
			refString = strings.TrimPrefix(refString, v2Prefix)
		} else {
			const v3Prefix = "#/components/schemas"
			refString = strings.TrimPrefix(refString, v3Prefix)
		}
	} else if protoRef, ok := model.(proto.Reference); ok {
		ext = protoRef.GetExtensions()
		refString = protoRef.Reference()
	} else if protoModels, ok := model.(proto.Schema); ok {
		ext = protoModels.GetExtensions()
	} else {
		c.reportError("unrecognized model type: %T", model)
	}

	if len(refString) > 0 {
		if refString == "io.k8s.apimachinery.pkg.runtime.RawExtension" {
			return schema.TypeRef{
				NamedType: &untypedName,
			}
		}
		// reference a named type
		_, n := path.Split(refString)
		tr.NamedType = &n

		//!TODO: Refactor the field ElementRelationship override
		// we can generate the types with overrides ahead of time rather than
		// requiring the hacky runtime support
		// (could just create a normalized key struct containing all customizations
		// 	to deduplicate)
		if mapRelationship, err := getMapElementRelationship(ext); err != nil {
			c.reportError(err.Error())
		} else if len(mapRelationship) > 0 {
			tr.ElementRelationship = &mapRelationship
		}
	} else {
		// compute the type inline
		c2 := c.push("inlined in "+c.currentName, &tr.Inlined)
		c2.preserveUnknownFields = preserveUnknownFields
		if protoModels, ok := model.(proto.Schema); ok {
			protoModels.Accept(c2)
		} else if specSchema, ok := model.(*spec.Schema); ok {
			c2.VisitSpec(specSchema)
		}
		c.pop(c2)

		if tr == (schema.TypeRef{}) {
			// emit warning?
			tr.NamedType = &untypedName
		} else if tr.Inlined == deducedDef.Atom {
			// Deduplicate deducedDef (mostly for testing)
			tr = schema.TypeRef{
				NamedType: &deducedName,
			}
		}
	}

	return tr
}

func makeUnions(extensions map[string]interface{}) ([]schema.Union, error) {
	schemaUnions := []schema.Union{}
	if iunions, ok := extensions["x-kubernetes-unions"]; ok {
		unions, ok := iunions.([]interface{})
		if !ok {
			return nil, fmt.Errorf(`"x-kubernetes-unions" should be a list, got %#v`, unions)
		}
		for _, iunion := range unions {
			var unionMap map[string]any
			unionMap, ok := iunion.(map[string]any)
			if !ok {
				union, ok := iunion.(map[interface{}]interface{})
				if !ok {
					return nil, fmt.Errorf(`"x-kubernetes-unions" items should be a map of string to unions, got %#v`, iunion)
				}
				unionMap = map[string]interface{}{}
				for k, v := range union {
					key, ok := k.(string)
					if !ok {
						return nil, fmt.Errorf(`"x-kubernetes-unions" has non-string key: %#v`, k)
					}
					unionMap[key] = v
				}
			}
			schemaUnion, err := makeUnion(unionMap)
			if err != nil {
				return nil, err
			}
			schemaUnions = append(schemaUnions, schemaUnion)
		}
	}

	// Make sure we have no overlap between unions
	fs := map[string]struct{}{}
	for _, u := range schemaUnions {
		if u.Discriminator != nil {
			if _, ok := fs[*u.Discriminator]; ok {
				return nil, fmt.Errorf("%v field appears multiple times in unions", *u.Discriminator)
			}
			fs[*u.Discriminator] = struct{}{}
		}
		for _, f := range u.Fields {
			if _, ok := fs[f.FieldName]; ok {
				return nil, fmt.Errorf("%v field appears multiple times in unions", f.FieldName)
			}
			fs[f.FieldName] = struct{}{}
		}
	}

	return schemaUnions, nil
}

func makeUnion(extensions map[string]interface{}) (schema.Union, error) {
	union := schema.Union{
		Fields: []schema.UnionField{},
	}

	if idiscriminator, ok := extensions["discriminator"]; ok {
		discriminator, ok := idiscriminator.(string)
		if !ok {
			return schema.Union{}, fmt.Errorf(`"discriminator" must be a string, got: %#v`, idiscriminator)
		}
		union.Discriminator = &discriminator
	}

	if ifields, ok := extensions["fields-to-discriminateBy"]; ok {
		fields, ok := ifields.(map[string]any)
		if !ok {
			ifields, ok := ifields.(map[interface{}]interface{})
			if !ok {
				return schema.Union{}, fmt.Errorf(`"fields-to-discriminateBy" must be a map[string]string, got: %#v`, ifields)
			}

			fields = map[string]any{}
			for ifield, val := range ifields {
				field, ok := ifield.(string)
				if !ok {
					return schema.Union{}, fmt.Errorf(`"fields-to-discriminateBy": field must be a string, got: %#v`, ifield)
				}
				fields[field] = val
			}
		}
		// Needs sorted keys by field.
		keys := []string{}
		for field := range fields {
			keys = append(keys, field)
		}

		sort.Strings(keys)
		reverseMap := map[string]struct{}{}
		for _, field := range keys {
			value := fields[field]
			discriminated, ok := value.(string)
			if !ok {
				return schema.Union{}, fmt.Errorf(`"fields-to-discriminateBy"/%v: value must be a string, got: %#v`, field, value)
			}
			union.Fields = append(union.Fields, schema.UnionField{
				FieldName:          field,
				DiscriminatorValue: discriminated,
			})

			// Check that we don't have the same discriminateBy multiple times.
			if _, ok := reverseMap[discriminated]; ok {
				return schema.Union{}, fmt.Errorf("Multiple fields have the same discriminated name: %v", discriminated)
			}
			reverseMap[discriminated] = struct{}{}
		}
	}

	if union.Discriminator != nil && len(union.Fields) == 0 {
		return schema.Union{}, fmt.Errorf("discriminator set to %v, but no fields in union", *union.Discriminator)
	}
	return union, nil
}

func (c *convert) VisitKind(k *proto.Kind) {
	preserveUnknownFields := c.preserveUnknownFields
	if p, ok := k.GetExtensions()["x-kubernetes-preserve-unknown-fields"]; ok && p == true {
		preserveUnknownFields = true
	}

	a := c.top()
	a.Map = &schema.Map{}
	for _, name := range k.FieldOrder {
		member := k.Fields[name]
		tr := c.makeRef(member, preserveUnknownFields)
		a.Map.Fields = append(a.Map.Fields, schema.StructField{
			Name:    name,
			Type:    tr,
			Default: member.GetDefault(),
		})
	}

	unions, err := makeUnions(k.GetExtensions())
	if err != nil {
		c.reportError(err.Error())
		return
	}
	// TODO: We should check that the fields and discriminator
	// specified in the union are actual fields in the struct.
	a.Map.Unions = unions

	if preserveUnknownFields {
		a.Map.ElementType = schema.TypeRef{
			NamedType: &deducedName,
		}
	}

	if mapRelationship, err := getMapElementRelationship(k.GetExtensions()); err != nil {
		c.reportError(err.Error())
	} else if len(mapRelationship) > 0 {
		a.Map.ElementRelationship = mapRelationship
	}
}

func (c *convert) VisitArray(a *proto.Array) {
	converted, err := convertArray(a.GetExtensions(), c.makeRef(a.SubType, c.preserveUnknownFields))
	if err != nil {
		c.reportError(err.Error())
	} else {
		*c.top() = converted
	}
}

func (c *convert) VisitMap(m *proto.Map) {
	a := c.top()
	a.Map = &schema.Map{}
	a.Map.ElementType = c.makeRef(m.SubType, c.preserveUnknownFields)

	if mapRelationship, err := getMapElementRelationship(m.GetExtensions()); err != nil {
		c.reportError(err.Error())
	} else if len(mapRelationship) > 0 {
		a.Map.ElementRelationship = mapRelationship
	}
}

func ptr(s schema.Scalar) *schema.Scalar { return &s }

func (c *convert) VisitPrimitive(p *proto.Primitive) {
	a := c.top()
	if c.currentName == quantityResource {
		a.Scalar = ptr(schema.Scalar("untyped"))
	} else {
		*a = convertPrimitive(p.Type, p.Format)
	}
}
func (c *convert) VisitArbitrary(a *proto.Arbitrary) {
	*c.top() = deducedDef.Atom
}

func (c *convert) VisitReference(proto.Reference) {
	// Do nothing, we handle references specially
}

// Returns map element relationship if specified, or empty string if unspecified
func getMapElementRelationship(ext map[string]any) (schema.ElementRelationship, error) {
	val, ok := ext["x-kubernetes-map-type"]
	if !ok {
		return "", nil
	}

	switch val {
	case "atomic":
		return schema.Atomic, nil
	case "granular":
		return schema.Separable, nil
	default:
		return "", fmt.Errorf("unknown map type %v", val)
	}
}

// Returns array element relationship if specified, or empty string if unspecified
// Also returns the primary keys of the elements in the array, if applicable to the
// element relationship.
func getArrayElementRelationship(ext map[string]any) (schema.ElementRelationship, []string, error) {
	if val, ok := ext["x-kubernetes-list-type"]; ok {
		switch val {
		case "atomic":
			return schema.Atomic, nil, nil
		case "set":
			return schema.Associative, nil, nil
		case "map":
			if keys, ok := ext["x-kubernetes-list-map-keys"]; ok {
				if keyNames, ok := toStringSlice(keys); ok {
					return schema.Associative, keyNames, nil
				} else {
					return "", nil, fmt.Errorf("uninterpreted map keys: %#v", keys)
				}
			}

			return "", nil, fmt.Errorf("missing map keys")
		default:
			return "", nil, fmt.Errorf("unknown list type: %v", val)
		}
	} else if val, ok := ext["x-kubernetes-patch-strategy"]; ok {
		switch val {
		case "merge", "merge,retainKeys":
			if key, ok := ext["x-kubernetes-patch-merge-key"]; ok {
				keyName, ok := key.(string)
				if !ok {
					return "", nil, fmt.Errorf("uninterpreted merge key: %#v", key)
				}

				return schema.Associative, []string{keyName}, nil
			}

			// It's not an error for merge key to be absent, it
			// means it's a set.
			return schema.Associative, nil, nil
		case "retainKeys":
			return "", nil, nil
		default:
			return "", nil, fmt.Errorf("unknown patch strategy %v", val)
		}
	}

	return "", nil, nil
}
