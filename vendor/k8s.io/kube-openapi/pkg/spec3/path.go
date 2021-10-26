package spec3

import (
	"encoding/json"
	"strconv"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
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

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *Paths) UnmarshalJSON(data []byte) error {
	var res map[string]json.RawMessage
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	for k, v := range res {
		if strings.HasPrefix(strings.ToLower(k), "x-") {
			if p.Extensions == nil {
				p.Extensions = make(map[string]interface{})
			}
			var d interface{}
			if err := json.Unmarshal(v, &d); err != nil {
				return err
			}
			p.Extensions[k] = d
		}
		if strings.HasPrefix(k, "/") {
			if p.Paths == nil {
				p.Paths = make(map[string]*Path)
			}
			var pi *Path
			if err := json.Unmarshal(v, &pi); err != nil {
				return err
			}
			p.Paths[k] = pi
		}
	}
	return nil
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
	// Servers is an alternative server array to service all operations in this path
	Servers []*Server `json:"servers,omitempty"`
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

func (r *Responses) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &r.ResponsesProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}

	return nil
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

// UnmarshalJSON unmarshals responses from JSON
func (r *ResponsesProps) UnmarshalJSON(data []byte) error {
	var res map[string]*Response
	if err := json.Unmarshal(data, &res); err != nil {
		return nil
	}
	if v, ok := res["default"]; ok {
		r.Default = v
		delete(res, "default")
	}
	for k, v := range res {
		if nk, err := strconv.Atoi(k); err == nil {
			if r.StatusCodeResponses == nil {
				r.StatusCodeResponses = map[int]*Response{}
			}
			r.StatusCodeResponses[nk] = v
		}
	}
	return nil
}

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
	if err := json.Unmarshal(data, &e.ExternalDocumentationProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.VendorExtensible); err != nil {
		return err
	}
	return nil
}

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
	if err := json.Unmarshal(data, &s.ServerProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
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
	if err := json.Unmarshal(data, &s.ServerVariableProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
	return nil
}
