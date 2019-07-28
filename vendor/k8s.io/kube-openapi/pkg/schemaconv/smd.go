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
	"sigs.k8s.io/structured-merge-diff/schema"
)

// ToSchema converts openapi definitions into a schema suitable for structured
// merge (i.e. kubectl apply v2).
func ToSchema(models proto.Models) (*schema.Schema, error) {
	c := convert{
		input:  models,
		output: &schema.Schema{},
	}
	if err := c.convertAll(); err != nil {
		return nil, err
	}
	c.addCommonTypes()
	return c.output, nil
}

type convert struct {
	input  proto.Models
	output *schema.Schema

	currentName   string
	current       *schema.Atom
	errorMessages []string
}

func (c *convert) push(name string, a *schema.Atom) *convert {
	return &convert{
		input:       c.input,
		output:      c.output,
		currentName: name,
		current:     a,
	}
}

func (c *convert) top() *schema.Atom { return c.current }

func (c *convert) pop(c2 *convert) {
	c.errorMessages = append(c.errorMessages, c2.errorMessages...)
}

func (c *convert) convertAll() error {
	for _, name := range c.input.ListModels() {
		model := c.input.LookupModel(name)
		c.insertTypeDef(name, model)
	}
	if len(c.errorMessages) > 0 {
		return errors.New(strings.Join(c.errorMessages, "\n"))
	}
	return nil
}

func (c *convert) reportError(format string, args ...interface{}) {
	c.errorMessages = append(c.errorMessages,
		c.currentName+": "+fmt.Sprintf(format, args...),
	)
}

func (c *convert) insertTypeDef(name string, model proto.Schema) {
	def := schema.TypeDef{
		Name: name,
	}
	c2 := c.push(name, &def.Atom)
	model.Accept(c2)
	c.pop(c2)
	if def.Atom == (schema.Atom{}) {
		// This could happen if there were a top-level reference.
		return
	}
	c.output.Types = append(c.output.Types, def)
}

func (c *convert) addCommonTypes() {
	c.output.Types = append(c.output.Types, untypedDef)
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

func (c *convert) makeRef(model proto.Schema) schema.TypeRef {
	var tr schema.TypeRef
	if r, ok := model.(*proto.Ref); ok {
		if r.Reference() == "io.k8s.apimachinery.pkg.runtime.RawExtension" {
			return schema.TypeRef{
				NamedType: &untypedName,
			}
		}
		// reference a named type
		_, n := path.Split(r.Reference())
		tr.NamedType = &n
	} else {
		// compute the type inline
		c2 := c.push("inlined in "+c.currentName, &tr.Inlined)
		model.Accept(c2)
		c.pop(c2)

		if tr == (schema.TypeRef{}) {
			// emit warning?
			tr.NamedType = &untypedName
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
			union, ok := iunion.(map[interface{}]interface{})
			if !ok {
				return nil, fmt.Errorf(`"x-kubernetes-unions" items should be a map of string to unions, got %#v`, iunion)
			}
			unionMap := map[string]interface{}{}
			for k, v := range union {
				key, ok := k.(string)
				if !ok {
					return nil, fmt.Errorf(`"x-kubernetes-unions" has non-string key: %#v`, k)
				}
				unionMap[key] = v
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
		fields, ok := ifields.(map[interface{}]interface{})
		if !ok {
			return schema.Union{}, fmt.Errorf(`"fields-to-discriminateBy" must be a map[string]string, got: %#v`, ifields)
		}
		// Needs sorted keys by field.
		keys := []string{}
		for ifield := range fields {
			field, ok := ifield.(string)
			if !ok {
				return schema.Union{}, fmt.Errorf(`"fields-to-discriminateBy": field must be a string, got: %#v`, ifield)
			}
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
				FieldName:       field,
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
	a := c.top()
	a.Map = &schema.Map{}
	for _, name := range k.FieldOrder {
		member := k.Fields[name]
		tr := c.makeRef(member)
		a.Map.Fields = append(a.Map.Fields, schema.StructField{
			Name: name,
			Type: tr,
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

	// TODO: Get element relationship when we start adding it to the spec.
}

func toStringSlice(o interface{}) (out []string, ok bool) {
	switch t := o.(type) {
	case []interface{}:
		for _, v := range t {
			switch vt := v.(type) {
			case string:
				out = append(out, vt)
			}
		}
		return out, true
	}
	return nil, false
}

func (c *convert) VisitArray(a *proto.Array) {
	atom := c.top()
	atom.List = &schema.List{
		ElementRelationship: schema.Atomic,
	}
	l := atom.List
	l.ElementType = c.makeRef(a.SubType)

	ext := a.GetExtensions()

	if val, ok := ext["x-kubernetes-list-type"]; ok {
		if val == "atomic" {
			l.ElementRelationship = schema.Atomic
		} else if val == "set" {
			l.ElementRelationship = schema.Associative
		} else if val == "map" {
			l.ElementRelationship = schema.Associative
			if keys, ok := ext["x-kubernetes-list-map-keys"]; ok {
				if keyNames, ok := toStringSlice(keys); ok {
					l.Keys = keyNames
				} else {
					c.reportError("uninterpreted map keys: %#v", keys)
				}
			} else {
				c.reportError("missing map keys")
			}
		} else {
			c.reportError("unknown list type %v", val)
			l.ElementRelationship = schema.Atomic
		}
	} else if val, ok := ext["x-kubernetes-patch-strategy"]; ok {
		if val == "merge" || val == "merge,retainKeys" {
			l.ElementRelationship = schema.Associative
			if key, ok := ext["x-kubernetes-patch-merge-key"]; ok {
				if keyName, ok := key.(string); ok {
					l.Keys = []string{keyName}
				} else {
					c.reportError("uninterpreted merge key: %#v", key)
				}
			} else {
				// It's not an error for this to be absent, it
				// means it's a set.
			}
		} else if val == "retainKeys" {
		} else {
			c.reportError("unknown patch strategy %v", val)
			l.ElementRelationship = schema.Atomic
		}
	}
}

func (c *convert) VisitMap(m *proto.Map) {
	a := c.top()
	a.Map = &schema.Map{}
	a.Map.ElementType = c.makeRef(m.SubType)

	// TODO: Get element relationship when we start putting it into the
	// spec.
}

func ptr(s schema.Scalar) *schema.Scalar { return &s }

func (c *convert) VisitPrimitive(p *proto.Primitive) {
	a := c.top()
	switch p.Type {
	case proto.Integer:
		a.Scalar = ptr(schema.Numeric)
	case proto.Number:
		a.Scalar = ptr(schema.Numeric)
	case proto.String:
		switch p.Format {
		case "":
			a.Scalar = ptr(schema.String)
		case "byte":
			// byte really means []byte and is encoded as a string.
			a.Scalar = ptr(schema.String)
		case "int-or-string":
			a.Scalar = ptr(schema.Scalar("untyped"))
		case "date-time":
			a.Scalar = ptr(schema.Scalar("untyped"))
		default:
			a.Scalar = ptr(schema.Scalar("untyped"))
		}
	case proto.Boolean:
		a.Scalar = ptr(schema.Boolean)
	default:
		a.Scalar = ptr(schema.Scalar("untyped"))
	}
}

func (c *convert) VisitArbitrary(a *proto.Arbitrary) {
	*c.top() = untypedDef.Atom
}

func (c *convert) VisitReference(proto.Reference) {
	// Do nothing, we handle references specially
}
