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

package spec

import (
	"errors"
	"strconv"

	"github.com/go-openapi/jsonreference"
	openapi_v2 "github.com/google/gnostic-models/openapiv2"
)

// Interfaces
type GnosticCommonValidations interface {
	GetMaximum() float64
	GetExclusiveMaximum() bool
	GetMinimum() float64
	GetExclusiveMinimum() bool
	GetMaxLength() int64
	GetMinLength() int64
	GetPattern() string
	GetMaxItems() int64
	GetMinItems() int64
	GetUniqueItems() bool
	GetMultipleOf() float64
	GetEnum() []*openapi_v2.Any
}

func (k *CommonValidations) FromGnostic(g GnosticCommonValidations) error {
	if g == nil {
		return nil
	}

	max := g.GetMaximum()
	if max != 0 {
		k.Maximum = &max
	}

	k.ExclusiveMaximum = g.GetExclusiveMaximum()

	min := g.GetMinimum()
	if min != 0 {
		k.Minimum = &min
	}

	k.ExclusiveMinimum = g.GetExclusiveMinimum()

	maxLen := g.GetMaxLength()
	if maxLen != 0 {
		k.MaxLength = &maxLen
	}

	minLen := g.GetMinLength()
	if minLen != 0 {
		k.MinLength = &minLen
	}

	k.Pattern = g.GetPattern()

	maxItems := g.GetMaxItems()
	if maxItems != 0 {
		k.MaxItems = &maxItems
	}

	minItems := g.GetMinItems()
	if minItems != 0 {
		k.MinItems = &minItems
	}

	k.UniqueItems = g.GetUniqueItems()

	multOf := g.GetMultipleOf()
	if multOf != 0 {
		k.MultipleOf = &multOf
	}

	enums := g.GetEnum()

	if enums != nil {
		k.Enum = make([]interface{}, len(enums))
		for i, v := range enums {
			if v == nil {
				continue
			}

			var convert interface{}
			if err := v.ToRawInfo().Decode(&convert); err != nil {
				return err
			} else {
				k.Enum[i] = convert
			}
		}
	}

	return nil
}

type GnosticSimpleSchema interface {
	GetType() string
	GetFormat() string
	GetItems() *openapi_v2.PrimitivesItems
	GetCollectionFormat() string
	GetDefault() *openapi_v2.Any
}

func (k *SimpleSchema) FromGnostic(g GnosticSimpleSchema) error {
	if g == nil {
		return nil
	}

	k.Type = g.GetType()
	k.Format = g.GetFormat()
	k.CollectionFormat = g.GetCollectionFormat()

	items := g.GetItems()
	if items != nil {
		k.Items = &Items{}
		if err := k.Items.FromGnostic(items); err != nil {
			return err
		}
	}

	def := g.GetDefault()
	if def != nil {
		var convert interface{}
		if err := def.ToRawInfo().Decode(&convert); err != nil {
			return err
		} else {
			k.Default = convert
		}
	}

	return nil
}

func (k *Items) FromGnostic(g *openapi_v2.PrimitivesItems) error {
	if g == nil {
		return nil
	}

	if err := k.SimpleSchema.FromGnostic(g); err != nil {
		return err
	}

	if err := k.CommonValidations.FromGnostic(g); err != nil {
		return err
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return err
	}

	return nil
}

func (k *VendorExtensible) FromGnostic(g []*openapi_v2.NamedAny) error {
	if len(g) == 0 {
		return nil
	}

	k.Extensions = make(Extensions, len(g))
	for _, v := range g {
		if v == nil {
			continue
		}

		if v.Value == nil {
			k.Extensions[v.Name] = nil
			continue
		}

		var iface interface{}
		if err := v.Value.ToRawInfo().Decode(&iface); err != nil {
			return err
		} else {
			k.Extensions[v.Name] = iface
		}
	}
	return nil
}

func (k *Refable) FromGnostic(g string) error {
	return k.Ref.FromGnostic(g)
}

func (k *Ref) FromGnostic(g string) error {
	if g == "" {
		return nil
	}

	ref, err := jsonreference.New(g)
	if err != nil {
		return err
	}

	*k = Ref{
		Ref: ref,
	}

	return nil
}

// Converts a gnostic v2 Document to a kube-openapi Swagger Document
//
// Caveats:
//
// - gnostic v2 documents treats zero as unspecified for numerical fields of
// CommonValidations fields such as Maximum, Minimum, MaximumItems, etc.
// There will always be data loss if one of the values of these fields is set to zero.
//
// Returns:
//
// - `ok`: `false` if a value was present in the gnostic document which cannot be
// roundtripped into kube-openapi types. In these instances, `ok` is set to
// `false` and the value is skipped.
//
// - `err`: an unexpected error occurred in the conversion from the gnostic type
// to kube-openapi type.
func (k *Swagger) FromGnostic(g *openapi_v2.Document) (ok bool, err error) {
	ok = true
	if g == nil {
		return true, nil
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	if nok, err := k.SwaggerProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *SwaggerProps) FromGnostic(g *openapi_v2.Document) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	// openapi_v2.Document does not support "ID" field, so it will not be
	// included
	k.Consumes = g.Consumes
	k.Produces = g.Produces
	k.Schemes = g.Schemes
	k.Swagger = g.Swagger

	if g.Info != nil {
		k.Info = &Info{}
		if nok, err := k.Info.FromGnostic(g.Info); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	k.Host = g.Host
	k.BasePath = g.BasePath

	if g.Paths != nil {
		k.Paths = &Paths{}
		if nok, err := k.Paths.FromGnostic(g.Paths); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Definitions != nil {
		k.Definitions = make(Definitions, len(g.Definitions.AdditionalProperties))
		for _, v := range g.Definitions.AdditionalProperties {
			if v == nil {
				continue
			}
			converted := Schema{}
			if nok, err := converted.FromGnostic(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
			k.Definitions[v.Name] = converted

		}
	}

	if g.Parameters != nil {
		k.Parameters = make(
			map[string]Parameter,
			len(g.Parameters.AdditionalProperties))
		for _, v := range g.Parameters.AdditionalProperties {
			if v == nil {
				continue
			}
			p := Parameter{}
			if nok, err := p.FromGnostic(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			k.Parameters[v.Name] = p
		}
	}

	if g.Responses != nil {
		k.Responses = make(
			map[string]Response,
			len(g.Responses.AdditionalProperties))

		for _, v := range g.Responses.AdditionalProperties {
			if v == nil {
				continue
			}
			p := Response{}
			if nok, err := p.FromGnostic(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			k.Responses[v.Name] = p
		}
	}

	if g.SecurityDefinitions != nil {
		k.SecurityDefinitions = make(SecurityDefinitions)
		if err := k.SecurityDefinitions.FromGnostic(g.SecurityDefinitions); err != nil {
			return false, err
		}
	}

	if g.Security != nil {
		k.Security = make([]map[string][]string, len(g.Security))
		for i, v := range g.Security {
			if v == nil || v.AdditionalProperties == nil {
				continue
			}

			k.Security[i] = make(map[string][]string, len(v.AdditionalProperties))
			converted := k.Security[i]
			for _, p := range v.AdditionalProperties {
				if p == nil {
					continue
				}
				if p.Value != nil {
					converted[p.Name] = p.Value.Value
				} else {
					converted[p.Name] = nil
				}
			}
		}
	}

	if g.Tags != nil {
		k.Tags = make([]Tag, len(g.Tags))
		for i, v := range g.Tags {
			if v == nil {
				continue
			} else if nok, err := k.Tags[i].FromGnostic(v); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}
	}

	if g.ExternalDocs != nil {
		k.ExternalDocs = &ExternalDocumentation{}
		if nok, err := k.ExternalDocs.FromGnostic(g.ExternalDocs); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	return ok, nil
}

// Info

func (k *Info) FromGnostic(g *openapi_v2.Info) (ok bool, err error) {
	ok = true
	if g == nil {
		return true, nil
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	if nok, err := k.InfoProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *InfoProps) FromGnostic(g *openapi_v2.Info) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	k.Description = g.Description
	k.Title = g.Title
	k.TermsOfService = g.TermsOfService

	if g.Contact != nil {
		k.Contact = &ContactInfo{}

		if nok, err := k.Contact.FromGnostic(g.Contact); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.License != nil {
		k.License = &License{}
		if nok, err := k.License.FromGnostic(g.License); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	k.Version = g.Version
	return ok, nil
}

func (k *License) FromGnostic(g *openapi_v2.License) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	k.Name = g.Name
	k.URL = g.Url

	// License does not embed to VendorExtensible!
	// data loss from g.VendorExtension
	if len(g.VendorExtension) != 0 {
		ok = false
	}

	return ok, nil
}

func (k *ContactInfo) FromGnostic(g *openapi_v2.Contact) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	k.Name = g.Name
	k.URL = g.Url
	k.Email = g.Email

	// ContactInfo does not embed to VendorExtensible!
	// data loss from g.VendorExtension
	if len(g.VendorExtension) != 0 {
		ok = false
	}

	return ok, nil
}

// Paths

func (k *Paths) FromGnostic(g *openapi_v2.Paths) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	if g.Path != nil {
		k.Paths = make(map[string]PathItem, len(g.Path))
		for _, v := range g.Path {
			if v == nil {
				continue
			}

			converted := PathItem{}
			if nok, err := converted.FromGnostic(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			k.Paths[v.Name] = converted
		}
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}
	return ok, nil
}

func (k *PathItem) FromGnostic(g *openapi_v2.PathItem) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true

	if nok, err := k.PathItemProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	if err := k.Refable.FromGnostic(g.XRef); err != nil {
		return false, err
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}
	return ok, nil
}

func (k *PathItemProps) FromGnostic(g *openapi_v2.PathItem) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	if g.Get != nil {
		k.Get = &Operation{}
		if nok, err := k.Get.FromGnostic(g.Get); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Put != nil {
		k.Put = &Operation{}
		if nok, err := k.Put.FromGnostic(g.Put); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Post != nil {
		k.Post = &Operation{}
		if nok, err := k.Post.FromGnostic(g.Post); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Delete != nil {
		k.Delete = &Operation{}
		if nok, err := k.Delete.FromGnostic(g.Delete); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Options != nil {
		k.Options = &Operation{}
		if nok, err := k.Options.FromGnostic(g.Options); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Head != nil {
		k.Head = &Operation{}
		if nok, err := k.Head.FromGnostic(g.Head); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Patch != nil {
		k.Patch = &Operation{}
		if nok, err := k.Patch.FromGnostic(g.Patch); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Parameters != nil {
		k.Parameters = make([]Parameter, len(g.Parameters))
		for i, v := range g.Parameters {
			if v == nil {
				continue
			} else if nok, err := k.Parameters[i].FromGnosticParametersItem(v); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}
	}

	return ok, nil
}

func (k *Operation) FromGnostic(g *openapi_v2.Operation) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	if nok, err := k.OperationProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *OperationProps) FromGnostic(g *openapi_v2.Operation) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	k.Description = g.Description
	k.Consumes = g.Consumes
	k.Produces = g.Produces
	k.Schemes = g.Schemes
	k.Tags = g.Tags
	k.Summary = g.Summary

	if g.ExternalDocs != nil {
		k.ExternalDocs = &ExternalDocumentation{}
		if nok, err := k.ExternalDocs.FromGnostic(g.ExternalDocs); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	k.ID = g.OperationId
	k.Deprecated = g.Deprecated

	if g.Security != nil {
		k.Security = make([]map[string][]string, len(g.Security))
		for i, v := range g.Security {
			if v == nil || v.AdditionalProperties == nil {
				continue
			}

			k.Security[i] = make(map[string][]string, len(v.AdditionalProperties))
			converted := k.Security[i]
			for _, p := range v.AdditionalProperties {
				if p == nil {
					continue
				}

				if p.Value != nil {
					converted[p.Name] = p.Value.Value
				} else {
					converted[p.Name] = nil
				}
			}
		}
	}

	if g.Parameters != nil {
		k.Parameters = make([]Parameter, len(g.Parameters))
		for i, v := range g.Parameters {
			if v == nil {
				continue
			} else if nok, err := k.Parameters[i].FromGnosticParametersItem(v); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}
	}

	if g.Responses != nil {
		k.Responses = &Responses{}
		if nok, err := k.Responses.FromGnostic(g.Responses); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	return ok, nil
}

// Responses

func (k *Responses) FromGnostic(g *openapi_v2.Responses) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	if nok, err := k.ResponsesProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *ResponsesProps) FromGnostic(g *openapi_v2.Responses) (ok bool, err error) {
	if g == nil {
		return true, nil
	} else if g.ResponseCode == nil {
		return ok, nil
	}

	ok = true
	for _, v := range g.ResponseCode {
		if v == nil {
			continue
		}
		if v.Name == "default" {
			k.Default = &Response{}
			if nok, err := k.Default.FromGnosticResponseValue(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		} else if nk, err := strconv.Atoi(v.Name); err != nil {
			// This should actually never fail, unless gnostic struct was
			// manually/purposefully tampered with at runtime.
			// Gnostic's ParseDocument validates that all StatusCodeResponses
			// 	keys adhere to the following regex ^([0-9]{3})$|^(default)$
			ok = false
		} else {
			if k.StatusCodeResponses == nil {
				k.StatusCodeResponses = map[int]Response{}
			}

			res := Response{}
			if nok, err := res.FromGnosticResponseValue(v.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
			k.StatusCodeResponses[nk] = res
		}
	}

	return ok, nil
}

func (k *Response) FromGnostic(g *openapi_v2.Response) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	// Refable case handled in FromGnosticResponseValue

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	if nok, err := k.ResponseProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *Response) FromGnosticResponseValue(g *openapi_v2.ResponseValue) (ok bool, err error) {
	ok = true
	if ref := g.GetJsonReference(); ref != nil {
		k.Description = ref.Description

		if err := k.Refable.FromGnostic(ref.XRef); err != nil {
			return false, err
		}
	} else if nok, err := k.FromGnostic(g.GetResponse()); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

func (k *ResponseProps) FromGnostic(g *openapi_v2.Response) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	k.Description = g.Description

	if g.Schema != nil {
		k.Schema = &Schema{}
		if nok, err := k.Schema.FromGnosticSchemaItem(g.Schema); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Headers != nil {
		k.Headers = make(map[string]Header, len(g.Headers.AdditionalProperties))
		for _, v := range g.Headers.AdditionalProperties {
			if v == nil {
				continue
			}

			converted := Header{}
			if err := converted.FromGnostic(v.GetValue()); err != nil {
				return false, err
			}

			k.Headers[v.Name] = converted
		}
	}

	if g.Examples != nil {
		k.Examples = make(map[string]interface{}, len(g.Examples.AdditionalProperties))
		for _, v := range g.Examples.AdditionalProperties {
			if v == nil {
				continue
			} else if v.Value == nil {
				k.Examples[v.Name] = nil
				continue
			}

			var iface interface{}
			if err := v.Value.ToRawInfo().Decode(&iface); err != nil {
				return false, err
			} else {
				k.Examples[v.Name] = iface
			}
		}
	}

	return ok, nil
}

// Header

func (k *Header) FromGnostic(g *openapi_v2.Header) (err error) {
	if g == nil {
		return nil
	}

	if err := k.CommonValidations.FromGnostic(g); err != nil {
		return err
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return err
	}

	if err := k.SimpleSchema.FromGnostic(g); err != nil {
		return err
	}

	if err := k.HeaderProps.FromGnostic(g); err != nil {
		return err
	}

	return nil
}

func (k *HeaderProps) FromGnostic(g *openapi_v2.Header) error {
	if g == nil {
		return nil
	}

	// All other fields of openapi_v2.Header are handled by
	// the embeded fields, commonvalidations, etc.
	k.Description = g.Description
	return nil
}

// Parameters

func (k *Parameter) FromGnostic(g *openapi_v2.Parameter) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	switch p := g.Oneof.(type) {
	case *openapi_v2.Parameter_BodyParameter:
		if nok, err := k.ParamProps.FromGnostic(p.BodyParameter); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}

		if err := k.VendorExtensible.FromGnostic(p.BodyParameter.GetVendorExtension()); err != nil {
			return false, err
		}

		return ok, nil
	case *openapi_v2.Parameter_NonBodyParameter:
		switch nb := g.GetNonBodyParameter().Oneof.(type) {
		case *openapi_v2.NonBodyParameter_HeaderParameterSubSchema:
			if nok, err := k.ParamProps.FromGnostic(nb.HeaderParameterSubSchema); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			if err := k.SimpleSchema.FromGnostic(nb.HeaderParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.CommonValidations.FromGnostic(nb.HeaderParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.VendorExtensible.FromGnostic(nb.HeaderParameterSubSchema.GetVendorExtension()); err != nil {
				return false, err
			}

			return ok, nil
		case *openapi_v2.NonBodyParameter_FormDataParameterSubSchema:
			if nok, err := k.ParamProps.FromGnostic(nb.FormDataParameterSubSchema); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			if err := k.SimpleSchema.FromGnostic(nb.FormDataParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.CommonValidations.FromGnostic(nb.FormDataParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.VendorExtensible.FromGnostic(nb.FormDataParameterSubSchema.GetVendorExtension()); err != nil {
				return false, err
			}

			return ok, nil
		case *openapi_v2.NonBodyParameter_QueryParameterSubSchema:
			if nok, err := k.ParamProps.FromGnostic(nb.QueryParameterSubSchema); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			if err := k.SimpleSchema.FromGnostic(nb.QueryParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.CommonValidations.FromGnostic(nb.QueryParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.VendorExtensible.FromGnostic(nb.QueryParameterSubSchema.GetVendorExtension()); err != nil {
				return false, err
			}

			return ok, nil
		case *openapi_v2.NonBodyParameter_PathParameterSubSchema:
			if nok, err := k.ParamProps.FromGnostic(nb.PathParameterSubSchema); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			if err := k.SimpleSchema.FromGnostic(nb.PathParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.CommonValidations.FromGnostic(nb.PathParameterSubSchema); err != nil {
				return false, err
			}

			if err := k.VendorExtensible.FromGnostic(nb.PathParameterSubSchema.GetVendorExtension()); err != nil {
				return false, err
			}

			return ok, nil
		default:
			return false, errors.New("unrecognized nonbody type for Parameter")
		}
	default:
		return false, errors.New("unrecognized type for Parameter")
	}
}

type GnosticCommonParamProps interface {
	GetName() string
	GetRequired() bool
	GetIn() string
	GetDescription() string
}

type GnosticCommonParamPropsBodyParameter interface {
	GetSchema() *openapi_v2.Schema
}

type GnosticCommonParamPropsFormData interface {
	GetAllowEmptyValue() bool
}

func (k *ParamProps) FromGnostic(g GnosticCommonParamProps) (ok bool, err error) {
	ok = true
	k.Description = g.GetDescription()
	k.In = g.GetIn()
	k.Name = g.GetName()
	k.Required = g.GetRequired()

	if formDataParameter, success := g.(GnosticCommonParamPropsFormData); success {
		k.AllowEmptyValue = formDataParameter.GetAllowEmptyValue()
	}

	if bodyParameter, success := g.(GnosticCommonParamPropsBodyParameter); success {
		if bodyParameter.GetSchema() != nil {
			k.Schema = &Schema{}
			if nok, err := k.Schema.FromGnostic(bodyParameter.GetSchema()); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}
	}

	return ok, nil
}

// PB types use a different structure than we do for "refable". For PB, there is
// a wrappign oneof type that could be a ref or the type
func (k *Parameter) FromGnosticParametersItem(g *openapi_v2.ParametersItem) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true
	if ref := g.GetJsonReference(); ref != nil {
		k.Description = ref.Description

		if err := k.Refable.FromGnostic(ref.XRef); err != nil {
			return false, err
		}
	} else if nok, err := k.FromGnostic(g.GetParameter()); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	return ok, nil
}

// Schema

func (k *Schema) FromGnostic(g *openapi_v2.Schema) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}

	// SwaggerSchemaProps
	k.Discriminator = g.Discriminator
	k.ReadOnly = g.ReadOnly
	k.Description = g.Description
	if g.ExternalDocs != nil {
		k.ExternalDocs = &ExternalDocumentation{}
		if nok, err := k.ExternalDocs.FromGnostic(g.ExternalDocs); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Example != nil {
		if err := g.Example.ToRawInfo().Decode(&k.Example); err != nil {
			return false, err
		}
	}

	// SchemaProps
	if err := k.Ref.FromGnostic(g.XRef); err != nil {
		return false, err
	}
	k.Type = g.Type.GetValue()
	k.Format = g.GetFormat()
	k.Title = g.GetTitle()

	// These below fields are not available in gnostic types, so will never
	// be populated. This means roundtrips which make use of these
	//	(non-official, kube-only) fields will lose information.
	//
	// Schema.ID is not available in official spec
	// Schema.$schema
	// Schema.Nullable - in openapiv3, not v2
	// Schema.AnyOf - in openapiv3, not v2
	// Schema.OneOf - in openapiv3, not v2
	// Schema.Not - in openapiv3, not v2
	// Schema.PatternProperties - in openapiv3, not v2
	// Schema.Dependencies - in openapiv3, not v2
	// Schema.AdditionalItems
	// Schema.Definitions - not part of spec
	// Schema.ExtraProps - gnostic parser rejects any keys it does not recognize

	if g.GetDefault() != nil {
		if err := g.GetDefault().ToRawInfo().Decode(&k.Default); err != nil {
			return false, err
		}
	}

	// These conditionals (!= 0) follow gnostic's logic for ToRawInfo
	// The keys in gnostic source are only included if nonzero.

	if g.Maximum != 0.0 {
		k.Maximum = &g.Maximum
	}

	if g.Minimum != 0.0 {
		k.Minimum = &g.Minimum
	}

	k.ExclusiveMaximum = g.ExclusiveMaximum
	k.ExclusiveMinimum = g.ExclusiveMinimum

	if g.MaxLength != 0 {
		k.MaxLength = &g.MaxLength
	}

	if g.MinLength != 0 {
		k.MinLength = &g.MinLength
	}

	k.Pattern = g.GetPattern()

	if g.MaxItems != 0 {
		k.MaxItems = &g.MaxItems
	}

	if g.MinItems != 0 {
		k.MinItems = &g.MinItems
	}
	k.UniqueItems = g.UniqueItems

	if g.MultipleOf != 0 {
		k.MultipleOf = &g.MultipleOf
	}

	for _, v := range g.GetEnum() {
		if v == nil {
			continue
		}

		var convert interface{}
		if err := v.ToRawInfo().Decode(&convert); err != nil {
			return false, err
		}
		k.Enum = append(k.Enum, convert)
	}

	if g.MaxProperties != 0 {
		k.MaxProperties = &g.MaxProperties
	}

	if g.MinProperties != 0 {
		k.MinProperties = &g.MinProperties
	}

	k.Required = g.Required

	if g.GetItems() != nil {
		k.Items = &SchemaOrArray{}
		for _, v := range g.Items.GetSchema() {
			if v == nil {
				continue
			}

			schema := Schema{}
			if nok, err := schema.FromGnostic(v); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
			k.Items.Schemas = append(k.Items.Schemas, schema)
		}

		if len(k.Items.Schemas) == 1 {
			k.Items.Schema = &k.Items.Schemas[0]
			k.Items.Schemas = nil
		}
	}

	for i, v := range g.GetAllOf() {
		if v == nil {
			continue
		}

		k.AllOf = append(k.AllOf, Schema{})
		if nok, err := k.AllOf[i].FromGnostic(v); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	if g.Properties != nil {
		k.Properties = make(map[string]Schema)
		for _, namedSchema := range g.Properties.AdditionalProperties {
			if namedSchema == nil {
				continue
			}
			val := &Schema{}
			if nok, err := val.FromGnostic(namedSchema.Value); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}

			k.Properties[namedSchema.Name] = *val
		}
	}

	if g.AdditionalProperties != nil {
		k.AdditionalProperties = &SchemaOrBool{}
		if g.AdditionalProperties.GetSchema() == nil {
			k.AdditionalProperties.Allows = g.AdditionalProperties.GetBoolean()
		} else {
			k.AdditionalProperties.Schema = &Schema{}
			k.AdditionalProperties.Allows = true

			if nok, err := k.AdditionalProperties.Schema.FromGnostic(g.AdditionalProperties.GetSchema()); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}
	}

	return ok, nil
}

func (k *Schema) FromGnosticSchemaItem(g *openapi_v2.SchemaItem) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true

	switch p := g.Oneof.(type) {
	case *openapi_v2.SchemaItem_FileSchema:
		fileSchema := p.FileSchema

		if err := k.VendorExtensible.FromGnostic(fileSchema.VendorExtension); err != nil {
			return false, err
		}

		k.Format = fileSchema.Format
		k.Title = fileSchema.Title
		k.Description = fileSchema.Description
		k.Required = fileSchema.Required
		k.Type = []string{fileSchema.Type}
		k.ReadOnly = fileSchema.ReadOnly

		if fileSchema.ExternalDocs != nil {
			k.ExternalDocs = &ExternalDocumentation{}
			if nok, err := k.ExternalDocs.FromGnostic(fileSchema.ExternalDocs); err != nil {
				return false, err
			} else if !nok {
				ok = false
			}
		}

		if fileSchema.Example != nil {
			if err := fileSchema.Example.ToRawInfo().Decode(&k.Example); err != nil {
				return false, err
			}
		}

		if fileSchema.Default != nil {
			if err := fileSchema.Default.ToRawInfo().Decode(&k.Default); err != nil {
				return false, err
			}
		}

	case *openapi_v2.SchemaItem_Schema:
		schema := p.Schema

		if nok, err := k.FromGnostic(schema); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	default:
		return false, errors.New("unrecognized type for SchemaItem")
	}

	return ok, nil
}

// SecurityDefinitions

func (k SecurityDefinitions) FromGnostic(g *openapi_v2.SecurityDefinitions) error {
	for _, v := range g.GetAdditionalProperties() {
		if v == nil {
			continue
		}
		secScheme := &SecurityScheme{}
		if err := secScheme.FromGnostic(v.Value); err != nil {
			return err
		}
		k[v.Name] = secScheme
	}

	return nil
}

type GnosticCommonSecurityDefinition interface {
	GetType() string
	GetDescription() string
}

func (k *SecuritySchemeProps) FromGnostic(g GnosticCommonSecurityDefinition) error {
	k.Type = g.GetType()
	k.Description = g.GetDescription()

	if hasName, success := g.(interface{ GetName() string }); success {
		k.Name = hasName.GetName()
	}

	if hasIn, success := g.(interface{ GetIn() string }); success {
		k.In = hasIn.GetIn()
	}

	if hasFlow, success := g.(interface{ GetFlow() string }); success {
		k.Flow = hasFlow.GetFlow()
	}

	if hasAuthURL, success := g.(interface{ GetAuthorizationUrl() string }); success {
		k.AuthorizationURL = hasAuthURL.GetAuthorizationUrl()
	}

	if hasTokenURL, success := g.(interface{ GetTokenUrl() string }); success {
		k.TokenURL = hasTokenURL.GetTokenUrl()
	}

	if hasScopes, success := g.(interface {
		GetScopes() *openapi_v2.Oauth2Scopes
	}); success {
		scopes := hasScopes.GetScopes()
		if scopes != nil {
			k.Scopes = make(map[string]string, len(scopes.AdditionalProperties))
			for _, v := range scopes.AdditionalProperties {
				if v == nil {
					continue
				}

				k.Scopes[v.Name] = v.Value
			}
		}
	}

	return nil
}

func (k *SecurityScheme) FromGnostic(g *openapi_v2.SecurityDefinitionsItem) error {
	if g == nil {
		return nil
	}

	switch s := g.Oneof.(type) {
	case *openapi_v2.SecurityDefinitionsItem_ApiKeySecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.ApiKeySecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.ApiKeySecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	case *openapi_v2.SecurityDefinitionsItem_BasicAuthenticationSecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.BasicAuthenticationSecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.BasicAuthenticationSecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	case *openapi_v2.SecurityDefinitionsItem_Oauth2AccessCodeSecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.Oauth2AccessCodeSecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.Oauth2AccessCodeSecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	case *openapi_v2.SecurityDefinitionsItem_Oauth2ApplicationSecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.Oauth2ApplicationSecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.Oauth2ApplicationSecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	case *openapi_v2.SecurityDefinitionsItem_Oauth2ImplicitSecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.Oauth2ImplicitSecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.Oauth2ImplicitSecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	case *openapi_v2.SecurityDefinitionsItem_Oauth2PasswordSecurity:
		if err := k.SecuritySchemeProps.FromGnostic(s.Oauth2PasswordSecurity); err != nil {
			return err
		}
		if err := k.VendorExtensible.FromGnostic(s.Oauth2PasswordSecurity.VendorExtension); err != nil {
			return err
		}
		return nil
	default:
		return errors.New("unrecognized SecurityDefinitionsItem")
	}
}

// Tag

func (k *Tag) FromGnostic(g *openapi_v2.Tag) (ok bool, err error) {
	if g == nil {
		return true, nil
	}

	ok = true

	if nok, err := k.TagProps.FromGnostic(g); err != nil {
		return false, err
	} else if !nok {
		ok = false
	}

	if err := k.VendorExtensible.FromGnostic(g.VendorExtension); err != nil {
		return false, err
	}
	return ok, nil
}

func (k *TagProps) FromGnostic(g *openapi_v2.Tag) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	k.Description = g.Description
	k.Name = g.Name

	if g.ExternalDocs != nil {
		k.ExternalDocs = &ExternalDocumentation{}
		if nok, err := k.ExternalDocs.FromGnostic(g.ExternalDocs); err != nil {
			return false, err
		} else if !nok {
			ok = false
		}
	}

	return ok, nil
}

// ExternalDocumentation

func (k *ExternalDocumentation) FromGnostic(g *openapi_v2.ExternalDocs) (ok bool, err error) {
	if g == nil {
		return true, nil
	}
	ok = true
	k.Description = g.Description
	k.URL = g.Url

	// data loss! g.VendorExtension
	if len(g.VendorExtension) != 0 {
		ok = false
	}

	return ok, nil
}
