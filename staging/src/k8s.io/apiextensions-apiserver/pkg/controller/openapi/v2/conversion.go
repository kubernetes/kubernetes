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

package v2

import (
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// ToStructuralOpenAPIV2 converts our internal OpenAPI v3 structural schema to
// to a v2 compatible schema.
func ToStructuralOpenAPIV2(in *structuralschema.Structural) *structuralschema.Structural {
	if in == nil {
		return nil
	}

	out := in.DeepCopy()

	// Remove unsupported fields in OpenAPI v2 recursively
	mapper := structuralschema.Visitor{
		Structural: func(s *structuralschema.Structural) bool {
			changed := false
			if s.ValueValidation != nil {
				if s.ValueValidation.AllOf != nil {
					s.ValueValidation.AllOf = nil
					changed = true
				}
				if s.ValueValidation.OneOf != nil {
					s.ValueValidation.OneOf = nil
					changed = true
				}
				if s.ValueValidation.AnyOf != nil {
					s.ValueValidation.AnyOf = nil
					changed = true
				}
				if s.ValueValidation.Not != nil {
					s.ValueValidation.Not = nil
					changed = true
				}
			}

			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-ce77fea74b9dd098045004410023e0c3R219
			if s.Nullable {
				s.Type = ""
				s.Nullable = false

				// untyped values break if items or properties are set in kubectl
				// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R183
				s.Items = nil
				s.Properties = nil

				changed = true
			}

			if s.XPreserveUnknownFields {
				// unknown fields break if items or properties are set in kubectl
				s.Items = nil
				s.Properties = nil

				changed = true
			}

			if s.Items == nil && s.Type == "array" {
				// kubectl cannot cope with array without item schema, e.g. due to XPreserveUnknownFields case above
				// https://github.com/kubernetes/kube-openapi/blob/64514a1d5d596b96e6f957e2be275ae14d6b0804/pkg/util/proto/document.go#L185
				s.Type = ""

				changed = true
			}

			for f, fs := range s.Properties {
				if fs.Nullable {
					s.ValueValidation.Required, changed = filterOut(s.ValueValidation.Required, f)
				}
			}
			if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil && s.AdditionalProperties.Structural.Nullable {
				s.ValueValidation.Required, changed = nil, true
			}

			return changed
		},
		// we drop all junctors above, and hence, never reach nested value validations
		NestedValueValidation: nil,
	}
	mapper.Visit(out)

	return out
}

func filterOut(ss []string, x string) ([]string, bool) {
	var filtered []string
	for i, s := range ss {
		if s == x {
			if filtered == nil {
				filtered = make([]string, i, len(ss))
				copy(filtered, ss[:i])
			}
		} else if filtered != nil {
			filtered = append(filtered, s)
		}
	}
	if filtered != nil {
		if len(filtered) == 0 {
			return nil, true
		}
		return filtered, true
	}
	return ss, false
}
