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

import "k8s.io/kube-openapi/pkg/validation/spec"

// Components holds a set of reusable objects for different aspects of the OAS.
// All objects defined within the components object will have no effect on the API
// unless they are explicitly referenced from properties outside the components object.
//
// more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#componentsObject
type Components struct {
	// Schemas holds reusable Schema Objects
	Schemas map[string]*spec.Schema `json:"schemas,omitempty"`
	// SecuritySchemes holds reusable Security Scheme Objects, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#securitySchemeObject
	SecuritySchemes SecuritySchemes `json:"securitySchemes,omitempty"`
	// Responses holds reusable Responses Objects
	Responses map[string]*Response `json:"responses,omitempty"`
	// Parameters holds reusable Parameters Objects
	Parameters map[string]*Parameter `json:"parameters,omitempty"`
	// Example holds reusable Example objects
	Examples map[string]*Example `json:"examples,omitempty"`
	// RequestBodies holds reusable Request Body objects
	RequestBodies map[string]*RequestBody `json:"requestBodies,omitempty"`
	// Links is a map of operations links that can be followed from the response
	Links map[string]*Link `json:"links,omitempty"`
	// Headers holds a maps of a headers name to its definition
	Headers map[string]*Header `json:"headers,omitempty"`
	// all fields are defined at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#componentsObject
}

// SecuritySchemes holds reusable Security Scheme Objects, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#securitySchemeObject
type SecuritySchemes map[string]*SecurityScheme
