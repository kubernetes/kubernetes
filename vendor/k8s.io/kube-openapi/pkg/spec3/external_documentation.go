/*
Copyright 2021 The Kubernetes Authors.

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

package spec3

import (
	"encoding/json"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type ExternalDocumentation struct {
	ExternalDocumentationProps
	spec.VendorExtensible
}

type ExternalDocumentationProps struct {
	// Description is a short description of the target documentation. CommonMark syntax MAY be used for rich text representation.
	Description string `json:"description,omitempty"`
	// URL is the URL for the target documentation.
	URL string `json:"url"`
}

// MarshalJSON is a custom marshal function that knows how to encode Responses as JSON
func (e *ExternalDocumentation) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(e.ExternalDocumentationProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(e.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (e *ExternalDocumentation) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, e)
	}
	if err := json.Unmarshal(data, &e.ExternalDocumentationProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (e *ExternalDocumentation) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		ExternalDocumentationProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	e.Extensions = internal.SanitizeExtensions(x.Extensions)
	e.ExternalDocumentationProps = x.ExternalDocumentationProps
	return nil
}
