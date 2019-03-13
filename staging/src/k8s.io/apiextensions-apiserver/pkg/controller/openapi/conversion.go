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

package openapi

import (
	"strings"

	"github.com/go-openapi/spec"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
)

// ConvertJSONSchemaPropsToOpenAPIv2Schema converts our internal OpenAPI v3 schema
// (*apiextensions.JSONSchemaProps) to an OpenAPI v2 schema (*spec.Schema).
func ConvertJSONSchemaPropsToOpenAPIv2Schema(in *apiextensions.JSONSchemaProps) (*spec.Schema, error) {
	if in == nil {
		return nil, nil
	}

	// dirty hack to temporarily set the type at the root. See continuation at the func bottom.
	// TODO: remove for Kubernetes 1.15
	oldRootType := in.Type
	if len(in.Type) == 0 {
		in.Type = "object"
	}

	// Remove unsupported fields in OpenAPI v2 recursively
	out := new(spec.Schema)
	validation.ConvertJSONSchemaPropsWithPostProcess(in, out, func(p *spec.Schema) error {
		p.OneOf = nil
		// TODO(roycaihw): preserve cases where we only have one subtree in AnyOf, same for OneOf
		p.AnyOf = nil
		p.Not = nil

		// TODO: drop everything below in 1.15 when we have passed one version skew towards kube-openapi in <1.14, which rejects valid openapi schemata

		if p.Ref.String() != "" {
			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R95
			p.Properties = nil

			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R99
			p.Type = nil

			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R104
			if !strings.HasPrefix(p.Ref.String(), "#/definitions/") {
				p.Ref = spec.Ref{}
			}
		}

		switch {
		case len(p.Type) == 2 && (p.Type[0] == "null" || p.Type[1] == "null"):
			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-ce77fea74b9dd098045004410023e0c3R219
			p.Type = nil
		case len(p.Type) == 1:
			switch p.Type[0] {
			case "null":
				// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-ce77fea74b9dd098045004410023e0c3R219
				p.Type = nil
			case "array":
				// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R183
				// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R184
				if p.Items == nil || (p.Items.Schema == nil && len(p.Items.Schemas) != 1) {
					p.Type = nil
					p.Items = nil
				}
			}
		case len(p.Type) > 1:
			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R272
			// We also set Properties to null to enforce parseArbitrary at https://github.com/kubernetes/kube-openapi/blob/814a8073653e40e0e324205d093770d4e7bb811f/pkg/util/proto/document.go#L247
			p.Type = nil
			p.Properties = nil
		default:
			// https://github.com/kubernetes/kube-openapi/pull/143/files#diff-62afddb578e9db18fb32ffb6b7802d92R248
			p.Properties = nil
		}

		// normalize items
		if p.Items != nil && len(p.Items.Schemas) == 1 {
			p.Items = &spec.SchemaOrArray{Schema: &p.Items.Schemas[0]}
		}

		// general fixups not supported by gnostic
		p.ID = ""
		p.Schema = ""
		p.Definitions = nil
		p.AdditionalItems = nil
		p.Dependencies = nil
		p.PatternProperties = nil
		if p.ExternalDocs != nil && len(p.ExternalDocs.URL) == 0 {
			p.ExternalDocs = nil
		}
		if p.Items != nil && p.Items.Schemas != nil {
			p.Items = nil
		}

		return nil
	})

	// restore root level type in input, and remove it in output if we had added it
	// TODO: remove with Kubernetes 1.15
	in.Type = oldRootType
	if len(oldRootType) == 0 {
		out.Type = nil
	}

	return out, nil
}
