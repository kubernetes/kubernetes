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

	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
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

func (c *convert) makeRef(model proto.Schema, preserveUnknownFields bool) schema.TypeRef {
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

		mapRelationship, err := getMapElementRelationship(model.GetExtensions())

		if err != nil {
			c.reportError(err.Error())
		}

		// empty string means unset.
		if len(mapRelationship) > 0 {
			tr.ElementRelationship = &mapRelationship
		}
	} else {
		// compute the type inline
		c2 := c.push("inlined in "+c.currentName, &tr.Inlined)
		c2.preserveUnknownFields = preserveUnknownFields
		model.Accept(c2)
		c.pop(c2)

		if tr == (schema.TypeRef{}) {
			// emit warning?
			tr.NamedType = &untypedName
		}
	}
	return tr
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

	a.Map.ElementRelationship, err = getMapElementRelationship(k.GetExtensions())
	if err != nil {
		c.reportError(err.Error())
	}
}

func (c *convert) VisitArray(a *proto.Array) {
	relationship, mapKeys, err := getListElementRelationship(a.GetExtensions())
	if err != nil {
		c.reportError(err.Error())
	}

	atom := c.top()
	atom.List = &schema.List{
		ElementType:         c.makeRef(a.SubType, c.preserveUnknownFields),
		ElementRelationship: relationship,
		Keys:                mapKeys,
	}
}

func (c *convert) VisitMap(m *proto.Map) {
	relationship, err := getMapElementRelationship(m.GetExtensions())
	if err != nil {
		c.reportError(err.Error())
	}

	a := c.top()
	a.Map = &schema.Map{
		ElementType:         c.makeRef(m.SubType, c.preserveUnknownFields),
		ElementRelationship: relationship,
	}
}

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
