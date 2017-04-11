// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"encoding/json"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/swag"
)

type OperationProps struct {
	Description  string                 `json:"description,omitempty"`
	Consumes     []string               `json:"consumes,omitempty"`
	Produces     []string               `json:"produces,omitempty"`
	Schemes      []string               `json:"schemes,omitempty"` // the scheme, when present must be from [http, https, ws, wss]
	Tags         []string               `json:"tags,omitempty"`
	Summary      string                 `json:"summary,omitempty"`
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitempty"`
	ID           string                 `json:"operationId,omitempty"`
	Deprecated   bool                   `json:"deprecated,omitempty"`
	Security     []map[string][]string  `json:"security,omitempty"`
	Parameters   []Parameter            `json:"parameters,omitempty"`
	Responses    *Responses             `json:"responses,omitempty"`
}

// Operation describes a single API operation on a path.
//
// For more information: http://goo.gl/8us55a#operationObject
type Operation struct {
	VendorExtensible
	OperationProps
}

// SuccessResponse gets a success response model
func (o *Operation) SuccessResponse() (*Response, int, bool) {
	if o.Responses == nil {
		return nil, 0, false
	}

	for k, v := range o.Responses.StatusCodeResponses {
		if k/100 == 2 {
			return &v, k, true
		}
	}

	return o.Responses.Default, 0, false
}

// JSONLookup look up a value by the json property name
func (o Operation) JSONLookup(token string) (interface{}, error) {
	if ex, ok := o.Extensions[token]; ok {
		return &ex, nil
	}
	r, _, err := jsonpointer.GetForToken(o.OperationProps, token)
	return r, err
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (o *Operation) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &o.OperationProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &o.VendorExtensible); err != nil {
		return err
	}
	return nil
}

// MarshalJSON converts this items object to JSON
func (o Operation) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(o.OperationProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(o.VendorExtensible)
	if err != nil {
		return nil, err
	}
	concated := swag.ConcatJSON(b1, b2)
	return concated, nil
}

// NewOperation creates a new operation instance.
// It expects an ID as parameter but not passing an ID is also valid.
func NewOperation(id string) *Operation {
	op := new(Operation)
	op.ID = id
	return op
}

// WithID sets the ID property on this operation, allows for chaining.
func (o *Operation) WithID(id string) *Operation {
	o.ID = id
	return o
}

// WithDescription sets the description on this operation, allows for chaining
func (o *Operation) WithDescription(description string) *Operation {
	o.Description = description
	return o
}

// WithSummary sets the summary on this operation, allows for chaining
func (o *Operation) WithSummary(summary string) *Operation {
	o.Summary = summary
	return o
}

// WithExternalDocs sets/removes the external docs for/from this operation.
// When you pass empty strings as params the external documents will be removed.
// When you pass non-empty string as one value then those values will be used on the external docs object.
// So when you pass a non-empty description, you should also pass the url and vice versa.
func (o *Operation) WithExternalDocs(description, url string) *Operation {
	if description == "" && url == "" {
		o.ExternalDocs = nil
		return o
	}

	if o.ExternalDocs == nil {
		o.ExternalDocs = &ExternalDocumentation{}
	}
	o.ExternalDocs.Description = description
	o.ExternalDocs.URL = url
	return o
}

// Deprecate marks the operation as deprecated
func (o *Operation) Deprecate() *Operation {
	o.Deprecated = true
	return o
}

// Undeprecate marks the operation as not deprected
func (o *Operation) Undeprecate() *Operation {
	o.Deprecated = false
	return o
}

// WithConsumes adds media types for incoming body values
func (o *Operation) WithConsumes(mediaTypes ...string) *Operation {
	o.Consumes = append(o.Consumes, mediaTypes...)
	return o
}

// WithProduces adds media types for outgoing body values
func (o *Operation) WithProduces(mediaTypes ...string) *Operation {
	o.Produces = append(o.Produces, mediaTypes...)
	return o
}

// WithTags adds tags for this operation
func (o *Operation) WithTags(tags ...string) *Operation {
	o.Tags = append(o.Tags, tags...)
	return o
}

// AddParam adds a parameter to this operation, when a parameter for that location
// and with that name already exists it will be replaced
func (o *Operation) AddParam(param *Parameter) *Operation {
	if param == nil {
		return o
	}

	for i, p := range o.Parameters {
		if p.Name == param.Name && p.In == param.In {
			params := append(o.Parameters[:i], *param)
			params = append(params, o.Parameters[i+1:]...)
			o.Parameters = params
			return o
		}
	}

	o.Parameters = append(o.Parameters, *param)
	return o
}

// RemoveParam removes a parameter from the operation
func (o *Operation) RemoveParam(name, in string) *Operation {
	for i, p := range o.Parameters {
		if p.Name == name && p.In == name {
			o.Parameters = append(o.Parameters[:i], o.Parameters[i+1:]...)
			return o
		}
	}
	return o
}

// SecuredWith adds a security scope to this operation.
func (o *Operation) SecuredWith(name string, scopes ...string) *Operation {
	o.Security = append(o.Security, map[string][]string{name: scopes})
	return o
}

// WithDefaultResponse adds a default response to the operation.
// Passing a nil value will remove the response
func (o *Operation) WithDefaultResponse(response *Response) *Operation {
	return o.RespondsWith(0, response)
}

// RespondsWith adds a status code response to the operation.
// When the code is 0 the value of the response will be used as default response value.
// When the value of the response is nil it will be removed from the operation
func (o *Operation) RespondsWith(code int, response *Response) *Operation {
	if o.Responses == nil {
		o.Responses = new(Responses)
	}
	if code == 0 {
		o.Responses.Default = response
		return o
	}
	if response == nil {
		delete(o.Responses.StatusCodeResponses, code)
		return o
	}
	if o.Responses.StatusCodeResponses == nil {
		o.Responses.StatusCodeResponses = make(map[int]Response)
	}
	o.Responses.StatusCodeResponses[code] = *response
	return o
}
