package spec3

import (
	"encoding/json"
	"strconv"

	"github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

// Paths describes the available paths and operations for the API, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathsObject
type Paths struct {
	Paths map[string]*Path
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Paths as JSON
func (p *Paths) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(p.Paths)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// Path describes the operations available on a single path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathItemObject
//
// Note that this struct is actually a thin wrapper around PathProps to make it referable and extensible
type Path struct {
	spec.Refable
	PathProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Path as JSON
func (p *Path) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(p.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(p.PathProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

// PathProps describes the operations available on a single path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathItemObject
type PathProps struct {
	// Summary holds a summary for all operations in this path
	Summary string `json:"summary,omitempty"`
	// Description holds a description for all operations in this path
	Description string `json:"description,omitempty"`
	// Get defines GET operation
	Get *Operation `json:"get,omitempty"`
	// Put defines PUT operation
	Put *Operation `json:"put,omitempty"`
	// Post defines POST operation
	Post *Operation `json:"post,omitempty"`
	// Delete defines DELETE operation
	Delete *Operation `json:"delete,omitempty"`
	// Options defines OPTIONS operation
	Options *Operation `json:"options,omitempty"`
	// Head defines HEAD operation
	Head *Operation `json:"head,omitempty"`
	// Patch defines PATCH operation
	Patch *Operation `json:"patch,omitempty"`
	// Trace defines TRACE operation
	Trace *Operation `json:"trace,omitempty"`
	// TODO: Servers field is missing - (servers	[Server Object]	An alternative server array to service all operations in this path.)
	// Parameters a list of parameters that are applicable for this operation
	Parameters []*Parameter `json:"parameters,omitempty"`
}

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

// OperationProps describes a single API operation on a path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#operationObject
type OperationProps struct {
	// Tags holds a list of tags for API documentation control
	Tags []string `json:"tags,omitempty"`
	// Summary holds a short summary of what the operation does
	Summary string `json:"summary,omitempty"`
	// Description holds a verbose explanation of the operation behavior
	Description string `json:"description,omitempty"`
	// TODO: ExternalDocs field is missing - (externalDocs External Documentation Object Additional external documentation for this operation)
	// OperationId holds a unique string used to identify the operation
	OperationId string `json:"operationId,omitempty"`
	// Parameters a list of parameters that are applicable for this operation
	// Parameters []*Parameter `json:"parameters,omitempty"`
	// RequestBody holds the request body applicable for this operation
	// RequestBody *RequestBody `json:"requestBody,omitempty"`
	// Responses holds the list of possible responses as they are returned from executing this operation
	Responses *Responses `json:"responses,omitempty"`
	// TODO: Callbacks field is missing - (callbacks	Map[string, Callback Object | Reference Object]	A map of possible out-of band callbacks related to the parent operation. The key is a unique identifier for the Callback Object. Each value in the map is a Callback Object that describes a request that may be initiated by the API provider and the expected responses. The key value used to identify the callback object is an expression, evaluated at runtime, that identifies a URL to use for the callback operation.
	// Deprecated declares this operation to be deprecated
	Deprecated bool `json:"deprecated,omitempty"`
	// TODO: Security field is missing - (security	[Security Requirement Object]	A declaration of which security mechanisms can be used for this operation. The list of values includes alternative security requirement objects that can be used. Only one of the security requirement objects need to be satisfied to authorize a request. This definition overrides any declared top-level security. To remove a top-level security declaration, an empty array can be used.)
	// TODO: Servers field is missing - (servers	[Server Object]	An alternative server array to service this operation. If an alternative server object is specified at the Path Item Object or Root level, it will be overridden by this value.)
}

// Responses holds the list of possible responses as they are returned from executing this operation
//
// Note that this struct is actually a thin wrapper around ResponsesProps to make it referable and extensible
type Responses struct {
	ResponsesProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Responses as JSON
func (r *Responses) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(r.ResponsesProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// ResponsesProps holds the list of possible responses as they are returned from executing this operation
type ResponsesProps struct {
	// Default holds the documentation of responses other than the ones declared for specific HTTP response codes. Use this field to cover undeclared responses
	Default *Response `json:"-"`
	// StatusCodeResponses holds a map of any HTTP status code to the response definition
	StatusCodeResponses map[int]*Response `json:"-"`
}

// MarshalJSON is a custom marshal function that knows how to encode ResponsesProps as JSON
func (r ResponsesProps) MarshalJSON() ([]byte, error) {
	toser := map[string]*Response{}
	if r.Default != nil {
		toser["default"] = r.Default
	}
	for k, v := range r.StatusCodeResponses {
		toser[strconv.Itoa(k)] = v
	}
	return json.Marshal(toser)
}
