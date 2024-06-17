/*
Copyright 2020 The Kubernetes Authors.

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
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// ToKubeOpenAPI converts a structural schema to go-openapi schema. It is faithful and roundtrippable.
func (s *Structural) ToKubeOpenAPI() *spec.Schema {
	if s == nil {
		return nil
	}

	ret := &spec.Schema{}

	if s.Items != nil {
		ret.Items = &spec.SchemaOrArray{Schema: s.Items.ToKubeOpenAPI()}
	}
	if s.Properties != nil {
		ret.Properties = make(map[string]spec.Schema, len(s.Properties))
		for k, v := range s.Properties {
			ret.Properties[k] = *v.ToKubeOpenAPI()
		}
	}
	if s.AdditionalProperties != nil {
		ret.AdditionalProperties = &spec.SchemaOrBool{
			Allows: s.AdditionalProperties.Bool,
			Schema: s.AdditionalProperties.Structural.ToKubeOpenAPI(),
		}
	}
	s.Generic.toKubeOpenAPI(ret)
	s.Extensions.toKubeOpenAPI(ret)
	s.ValueValidation.toKubeOpenAPI(ret)
	s.ValidationExtensions.toKubeOpenAPI(ret)
	return ret
}

func (g *Generic) toKubeOpenAPI(ret *spec.Schema) {
	if g == nil {
		return
	}

	if len(g.Type) != 0 {
		ret.Type = spec.StringOrArray{g.Type}
	}
	ret.Nullable = g.Nullable
	ret.Description = g.Description
	ret.Title = g.Title
	ret.Default = g.Default.Object
}

func (x *Extensions) toKubeOpenAPI(ret *spec.Schema) {
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
	if len(x.XListMapKeys) > 0 {
		ret.VendorExtensible.AddExtension("x-kubernetes-list-map-keys", x.XListMapKeys)
	}
	if x.XListType != nil {
		ret.VendorExtensible.AddExtension("x-kubernetes-list-type", *x.XListType)
	}
	if x.XMapType != nil {
		ret.VendorExtensible.AddExtension("x-kubernetes-map-type", *x.XMapType)
	}
}

func (x *ValidationExtensions) toKubeOpenAPI(ret *spec.Schema) {
	if x == nil {
		return
	}

	if len(x.XValidations) > 0 {
		ret.VendorExtensible.AddExtension("x-kubernetes-validations", x.XValidations)
	}
}

func (v *ValueValidation) toKubeOpenAPI(ret *spec.Schema) {
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
		ret.AllOf = append(ret.AllOf, *v.AllOf[i].toKubeOpenAPI())
	}
	for i := range v.AnyOf {
		ret.AnyOf = append(ret.AnyOf, *v.AnyOf[i].toKubeOpenAPI())
	}
	for i := range v.OneOf {
		ret.OneOf = append(ret.OneOf, *v.OneOf[i].toKubeOpenAPI())
	}
	ret.Not = v.Not.toKubeOpenAPI()
}

func (vv *NestedValueValidation) toKubeOpenAPI() *spec.Schema {
	if vv == nil {
		return nil
	}

	ret := &spec.Schema{}

	vv.ValueValidation.toKubeOpenAPI(ret)
	vv.ValidationExtensions.toKubeOpenAPI(ret)
	if vv.Items != nil {
		ret.Items = &spec.SchemaOrArray{Schema: vv.Items.toKubeOpenAPI()}
	}
	if vv.Properties != nil {
		ret.Properties = make(map[string]spec.Schema, len(vv.Properties))
		for k, v := range vv.Properties {
			ret.Properties[k] = *v.toKubeOpenAPI()
		}
	}
	vv.ForbiddenGenerics.toKubeOpenAPI(ret)   // normally empty. Exception: int-or-string
	vv.ForbiddenExtensions.toKubeOpenAPI(ret) // shouldn't do anything
	return ret
}
