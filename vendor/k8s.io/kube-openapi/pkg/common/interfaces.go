package common

// RouteContainer is the entrypoint for a service, which may contain multiple
// routes under a common path with a common set of path parameters.
type RouteContainer interface {
	// RootPath is the path that all contained routes are nested under.
	RootPath() string
	// PathParameters are common parameters defined in the root path.
	PathParameters() []Parameter
	// Routes are all routes exposed under the root path.
	Routes() []Route
}

// Route is a logical endpoint of a service.
type Route interface {
	// Method defines the HTTP Method.
	Method() string
	// Path defines the route's endpoint.
	Path() string
	// OperationName defines a machine-readable ID for the route.
	OperationName() string
	// Parameters defines the list of accepted parameters.
	Parameters() []Parameter
	// Description is a human-readable route description.
	Description() string
	// Consumes defines the consumed content-types.
	Consumes() []string
	// Produces defines the produced content-types.
	Produces() []string
	// Metadata allows adding extensions to the generated spec.
	Metadata() map[string]interface{}
	// RequestPayloadSample defines an example request payload. Can return nil.
	RequestPayloadSample() interface{}
	// ResponsePayloadSample defines an example response payload. Can return nil.
	ResponsePayloadSample() interface{}
	// StatusCodeResponses defines a mapping of HTTP Status Codes to the specific response(s).
	// Multiple responses with the same HTTP Status Code are acceptable.
	StatusCodeResponses() []StatusCodeResponse
}

// StatusCodeResponse is an explicit response type with an HTTP Status Code.
type StatusCodeResponse interface {
	// Code defines the HTTP Status Code.
	Code() int
	// Message returns the human-readable message.
	Message() string
	// Model defines an example payload for this response.
	Model() interface{}
}

// Parameter is a Route parameter.
type Parameter interface {
	// Name defines the unique-per-route identifier.
	Name() string
	// Description is the human-readable description of the param.
	Description() string
	// Required defines if this parameter must be provided.
	Required() bool
	// Kind defines the type of the parameter itself.
	Kind() ParameterKind
	// DataType defines the type of data the parameter carries.
	DataType() string
	// AllowMultiple defines if more than one value can be supplied for the parameter.
	AllowMultiple() bool
}

// ParameterKind is an enum of route parameter types.
type ParameterKind int

const (
	// PathParameterKind indicates the request parameter type is "path".
	PathParameterKind = ParameterKind(iota)

	// QueryParameterKind indicates the request parameter type is "query".
	QueryParameterKind

	// BodyParameterKind indicates the request parameter type is "body".
	BodyParameterKind

	// HeaderParameterKind indicates the request parameter type is "header".
	HeaderParameterKind

	// FormParameterKind indicates the request parameter type is "form".
	FormParameterKind

	// UnknownParameterKind indicates the request parameter type has not been specified.
	UnknownParameterKind
)
