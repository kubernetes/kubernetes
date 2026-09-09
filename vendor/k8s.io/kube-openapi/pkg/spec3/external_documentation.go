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
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	"k8s.io/kube-openapi/pkg/internal"
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
	return internal.DeterministicMarshal(e)
}

func (e *ExternalDocumentation) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		ExternalDocumentationProps `json:",embed"`
		Extensions                 spec.Extensions `json:",embed"`
	}
	x.Extensions = internal.SanitizeExtensions(e.Extensions)
	x.ExternalDocumentationProps = e.ExternalDocumentationProps
	return jsonv2.MarshalEncode(enc, x)
}

func (e *ExternalDocumentation) UnmarshalJSON(data []byte) error {
	return jsonv2.Unmarshal(data, e)
}

func (e *ExternalDocumentation) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		Extensions spec.Extensions `json:",embed"`
		ExternalDocumentationProps
	}
	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
		return err
	}
	e.Extensions = internal.SanitizeExtensions(x.Extensions)
	e.ExternalDocumentationProps = x.ExternalDocumentationProps
	return nil
}
