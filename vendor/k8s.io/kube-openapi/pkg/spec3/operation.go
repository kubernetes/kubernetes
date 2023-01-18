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
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// Operation describes a single API operation on a path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#operationObject
//
// Note that this struct is actually a thin wrapper around OperationProps to make it referable and extensible
type Operation struct {
	OperationProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Operation as JSON
func (o *Operation) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(o.OperationProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(o.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (o *Operation) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &o.OperationProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &o.VendorExtensible)
}

// OperationProps describes a single API operation on a path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#operationObject
type OperationProps struct {
	// Tags holds a list of tags for API documentation control
	Tags []string `json:"tags,omitempty"`
	// Summary holds a short summary of what the operation does
	Summary string `json:"summary,omitempty"`
	// Description holds a verbose explanation of the operation behavior
	Description string `json:"description,omitempty"`
	// ExternalDocs holds additional external documentation for this operation
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitempty"`
	// OperationId holds a unique string used to identify the operation
	OperationId string `json:"operationId,omitempty"`
	// Parameters a list of parameters that are applicable for this operation
	Parameters []*Parameter `json:"parameters,omitempty"`
	// RequestBody holds the request body applicable for this operation
	RequestBody *RequestBody `json:"requestBody,omitempty"`
	// Responses holds the list of possible responses as they are returned from executing this operation
	Responses *Responses `json:"responses,omitempty"`
	// Deprecated declares this operation to be deprecated
	Deprecated bool `json:"deprecated,omitempty"`
	// SecurityRequirement holds a declaration of which security mechanisms can be used for this operation
	SecurityRequirement []*SecurityRequirement `json:"security,omitempty"`
	// Servers contains an alternative server array to service this operation
	Servers []*Server `json:"servers,omitempty"`
}
