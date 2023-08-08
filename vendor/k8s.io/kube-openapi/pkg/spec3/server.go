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

type Server struct {
	ServerProps
	spec.VendorExtensible
}

type ServerProps struct {
	// Description is a short description of the target documentation. CommonMark syntax MAY be used for rich text representation.
	Description string `json:"description,omitempty"`
	// URL is the URL for the target documentation.
	URL string `json:"url"`
	// Variables contains a map between a variable name and its value. The value is used for substitution in the server's URL templeate
	Variables map[string]*ServerVariable `json:"variables,omitempty"`
}

// MarshalJSON is a custom marshal function that knows how to encode Responses as JSON
func (s *Server) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(s.ServerProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (s *Server) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, s)
	}

	if err := json.Unmarshal(data, &s.ServerProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (s *Server) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		ServerProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	s.Extensions = internal.SanitizeExtensions(x.Extensions)
	s.ServerProps = x.ServerProps

	return nil
}

type ServerVariable struct {
	ServerVariableProps
	spec.VendorExtensible
}

type ServerVariableProps struct {
	// Enum is an enumeration of string values to be used if the substitution options are from a limited set
	Enum []string `json:"enum,omitempty"`
	// Default is the default value to use for substitution, which SHALL be sent if an alternate value is not supplied
	Default string `json:"default"`
	// Description is a description for the server variable
	Description string `json:"description,omitempty"`
}

// MarshalJSON is a custom marshal function that knows how to encode Responses as JSON
func (s *ServerVariable) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(s.ServerVariableProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (s *ServerVariable) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, s)
	}
	if err := json.Unmarshal(data, &s.ServerVariableProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (s *ServerVariable) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		ServerVariableProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	s.Extensions = internal.SanitizeExtensions(x.Extensions)
	s.ServerVariableProps = x.ServerVariableProps

	return nil
}
