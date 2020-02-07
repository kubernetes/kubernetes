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

package smd

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"

	smdschema "sigs.k8s.io/structured-merge-diff/v3/schema"
)

// toAtom converts a structural schema to smd atom.
func toAtom(s *schema.Structural, preserveUnknown bool) (*smdschema.Atom, error) {
	if s.Extensions.XPreserveUnknownFields {
		preserveUnknown = true
	}
	switch s.Generic.Type {
	case "object":
		m := &smdschema.Map{
			Fields:              []smdschema.StructField{},
			ElementType:         smdschema.TypeRef{},
			ElementRelationship: smdschema.Separable,
		}
		if s.Properties != nil {
			for k, v := range s.Properties {
				a, err := toAtom(&v, preserveUnknown)
				if err != nil {
					return nil, err
				}
				m.Fields = append(m.Fields, smdschema.StructField{
					Name: k,
					Type: smdschema.TypeRef{
						Inlined: *a,
					},
				})
			}
			// TODO: Add unions once they are supported
		}
		if preserveUnknown {
			m.ElementType.NamedType = &deduced.Name
		}
		if s.Extensions.XMapType != nil {
			if *s.Extensions.XMapType == "atomic" {
				m.ElementRelationship = smdschema.Atomic
			} else if *s.Extensions.XMapType == "granular" {
				m.ElementRelationship = smdschema.Separable
			} else {
				return nil, fmt.Errorf("unknown map type %v", *s.Extensions.XMapType)
			}
		}
		if s.Extensions.XEmbeddedResource {
			m.Fields = mergeStructFields(baseResourceFields, m.Fields)
		}
		return &smdschema.Atom{
			Map: m,
		}, nil
	case "array":
		list := &smdschema.List{
			ElementType: smdschema.TypeRef{
				NamedType: &deduced.Name,
			},
			ElementRelationship: smdschema.Atomic,
		}
		if s.Items != nil {
			a, err := toAtom(s.Items, preserveUnknown)
			if err != nil {
				return nil, err
			}
			list.ElementType = smdschema.TypeRef{
				Inlined: *a,
			}
		}
		if s.Extensions.XListType != nil {
			if *s.Extensions.XListType == "atomic" {
				list.ElementRelationship = smdschema.Atomic
			} else if *s.Extensions.XListType == "set" {
				list.ElementRelationship = smdschema.Associative
			} else if *s.Extensions.XListType == "map" {
				if len(s.Extensions.XListMapKeys) > 0 {
					list.ElementRelationship = smdschema.Associative
					list.Keys = s.Extensions.XListMapKeys
				} else {
					return nil, fmt.Errorf("missing map keys")
				}
			} else {
				return nil, fmt.Errorf("unknown list type %v", *s.Extensions.XListType)
			}
		}
		return &smdschema.Atom{
			List: list,
		}, nil
	case "number":
		return &smdschema.Atom{Scalar: ptr(smdschema.Numeric)}, nil
	case "integer":
		return &smdschema.Atom{Scalar: ptr(smdschema.Numeric)}, nil
	case "boolean":
		return &smdschema.Atom{Scalar: ptr(smdschema.Boolean)}, nil
	case "string":
		return &smdschema.Atom{Scalar: ptr(smdschema.String)}, nil
	case "":
		if s.Extensions.XIntOrString {
			return &smdschema.Atom{Scalar: ptr(smdschema.Scalar("untyped"))}, nil
		} else if preserveUnknown {
			return &deduced.Atom, nil
		} else {
			return nil, fmt.Errorf("type is only optional if x-kubernetes-int-or-string or x-kubernetes-preserve-unknown-fields is true")
		}
	default:
		return nil, fmt.Errorf("type is not valid")
	}
}

func ptr(s smdschema.Scalar) *smdschema.Scalar { return &s }

func strPtr(s string) *string { return &s }

var deduced smdschema.TypeDef = smdschema.TypeDef{
	Name: "__untyped_deduced_",
	Atom: smdschema.Atom{
		Scalar: ptr(smdschema.Scalar("untyped")),
		List: &smdschema.List{
			ElementType: smdschema.TypeRef{
				NamedType: strPtr("__untyped_atomic_"),
			},
			ElementRelationship: smdschema.Atomic,
		},
		Map: &smdschema.Map{
			ElementType: smdschema.TypeRef{
				NamedType: strPtr("__untyped_deduced_"),
			},
			ElementRelationship: smdschema.Separable,
		},
	},
}

var baseResourceFields = []smdschema.StructField{
	{
		Name: "apiVersion",
		Type: smdschema.TypeRef{
			Inlined: smdschema.Atom{
				Scalar: ptr(smdschema.String),
			},
		},
	}, {
		Name: "kind",
		Type: smdschema.TypeRef{
			Inlined: smdschema.Atom{
				Scalar: ptr(smdschema.String),
			},
		},
	}, {
		Name: "metadata",
		Type: smdschema.TypeRef{
			NamedType: strPtr("io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"),
		},
	},
}

func mergeStructFields(lhs, rhs []smdschema.StructField) []smdschema.StructField {
	out := []smdschema.StructField{}
	lhsNames := map[string]struct{}{}
	for _, f := range lhs {
		lhsNames[f.Name] = struct{}{}
		out = append(out, f)
	}
	for _, f := range rhs {
		if _, ok := lhsNames[f.Name]; !ok {
			out = append(out, f)
		}
	}
	return out
}
