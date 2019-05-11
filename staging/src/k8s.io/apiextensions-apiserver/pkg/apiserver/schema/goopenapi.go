/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"github.com/go-openapi/spec"
)

// ToGoOpenAPI converts a structural schema to go-openapi schema. It is faithful and roundtrippable.
func (s *Structural) ToGoOpenAPI() *spec.Schema {
	if s == nil {
		return nil
	}

	ret := &spec.Schema{}

	if s.Items != nil {
		ret.Items = &spec.SchemaOrArray{Schema: s.Items.ToGoOpenAPI()}
	}
	if s.Properties != nil {
		ret.Properties = make(map[string]spec.Schema, len(s.Properties))
		for k, v := range s.Properties {
			ret.Properties[k] = *v.ToGoOpenAPI()
		}
	}
	s.Generic.toGoOpenAPI(ret)
	s.Extensions.toGoOpenAPI(ret)
	s.ValueValidation.toGoOpenAPI(ret)

	return ret
}

func (g *Generic) toGoOpenAPI(ret *spec.Schema) {
	if g == nil {
		return
	}

	if len(g.Type) != 0 {
		ret.Type = spec.StringOrArray{g.Type}
	}
	ret.Nullable = g.Nullable
	if g.AdditionalProperties != nil {
		ret.AdditionalProperties = &spec.SchemaOrBool{
			Allows: g.AdditionalProperties.Bool,
			Schema: g.AdditionalProperties.Structural.ToGoOpenAPI(),
		}
	}
	ret.Description = g.Description
	ret.Title = g.Title
	ret.Default = g.Default.Object
}

func (x *Extensions) toGoOpenAPI(ret *spec.Schema) {
	if x == nil {
		return
	}

	if x.XPreserveUnknownFields {
		ret.VendorExtensible.AddExtension("x-kubernetes-preserve-unknown-fields", true)
	}
	if x.XEmbeddedResource {
		ret.VendorExtensible.AddExtension("x-kubernetes-embedded-resource", true)
	}
	if x.XIntOrString {
		ret.VendorExtensible.AddExtension("x-kubernetes-int-or-string", true)
	}
}

func (v *ValueValidation) toGoOpenAPI(ret *spec.Schema) {
	if v == nil {
		return
	}

	ret.Format = v.Format
	ret.Maximum = v.Maximum
	ret.ExclusiveMaximum = v.ExclusiveMaximum
	ret.Minimum = v.Minimum
	ret.ExclusiveMinimum = v.ExclusiveMinimum
	ret.MaxLength = v.MaxLength
	ret.MinLength = v.MinLength
	ret.Pattern = v.Pattern
	ret.MaxItems = v.MaxItems
	ret.MinItems = v.MinItems
	ret.UniqueItems = v.UniqueItems
	ret.MultipleOf = v.MultipleOf
	if v.Enum != nil {
		ret.Enum = make([]interface{}, 0, len(v.Enum))
		for i := range v.Enum {
			ret.Enum = append(ret.Enum, v.Enum[i].Object)
		}
	}
	ret.MaxProperties = v.MaxProperties
	ret.MinProperties = v.MinProperties
	ret.Required = v.Required
	for i := range v.AllOf {
		ret.AllOf = append(ret.AllOf, *v.AllOf[i].toGoOpenAPI())
	}
	for i := range v.AnyOf {
		ret.AnyOf = append(ret.AnyOf, *v.AnyOf[i].toGoOpenAPI())
	}
	for i := range v.OneOf {
		ret.OneOf = append(ret.OneOf, *v.OneOf[i].toGoOpenAPI())
	}
	ret.Not = v.Not.toGoOpenAPI()
}

func (vv *NestedValueValidation) toGoOpenAPI() *spec.Schema {
	if vv == nil {
		return nil
	}

	ret := &spec.Schema{}

	vv.ValueValidation.toGoOpenAPI(ret)
	if vv.Items != nil {
		ret.Items = &spec.SchemaOrArray{Schema: vv.Items.toGoOpenAPI()}
	}
	if vv.Properties != nil {
		ret.Properties = make(map[string]spec.Schema, len(vv.Properties))
		for k, v := range vv.Properties {
			ret.Properties[k] = *v.toGoOpenAPI()
		}
	}
	vv.ForbiddenGenerics.toGoOpenAPI(ret)   // normally empty. Exception: int-or-string
	vv.ForbiddenExtensions.toGoOpenAPI(ret) // shouldn't do anything

	return ret
}
