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

// SecurityRequirementProps describes the required security schemes to execute an operation, more at https://swagger.io/specification/#security-requirement-object
//
// Note that this struct is actually a thin wrapper around SecurityRequirementProps to make it referable and extensible
type SecurityRequirement struct {
	SecurityRequirementProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode SecurityRequirement as JSON
func (s *SecurityRequirement) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(s.SecurityRequirementProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (s *SecurityRequirement) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &s.SecurityRequirementProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &s.VendorExtensible)
}

// SecurityRequirementProps describes the required security schemes to execute an operation, more at https://swagger.io/specification/#security-requirement-object
type SecurityRequirementProps map[string][]string
