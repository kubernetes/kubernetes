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

	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// OpenAPI is an object that describes an API and conforms to the OpenAPI Specification.
type OpenAPI struct {
	// Version represents the semantic version number of the OpenAPI Specification that this document uses
	Version string `json:"openapi"`
	// Info provides metadata about the API
	Info *spec.Info `json:"info"`
	// Paths holds the available target and operations for the API
	Paths *Paths `json:"paths,omitempty"`
	// Servers is an array of Server objects which provide connectivity information to a target server
	Servers []*Server `json:"servers,omitempty"`
	// Components hold various schemas for the specification
	Components *Components `json:"components,omitempty"`
	// ExternalDocs holds additional external documentation
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitempty"`
}

func (o *OpenAPI) UnmarshalJSON(data []byte) error {
	type OpenAPIWithNoFunctions OpenAPI
	p := (*OpenAPIWithNoFunctions)(o)
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, &p)
	}
	return json.Unmarshal(data, &p)
}
