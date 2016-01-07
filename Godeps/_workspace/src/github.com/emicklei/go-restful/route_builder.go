package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"os"
	"reflect"
	"runtime"
	"strings"

	"github.com/emicklei/go-restful/log"
)

// RouteBuilder is a helper to construct Routes.
type RouteBuilder struct {
	rootPath    string
	currentPath string
	produces    []string
	consumes    []string
	httpMethod  string        // required
	function    RouteFunction // required
	filters     []FilterFunction
	// documentation
	doc                     string
	notes                   string
	operation               string
	readSample, writeSample interface{}
	parameters              []*Parameter
	errorMap                map[int]ResponseError
}

// Do evaluates each argument with the RouteBuilder itself.
// This allows you to follow DRY principles without breaking the fluent programming style.
// Example:
// 		ws.Route(ws.DELETE("/{name}").To(t.deletePerson).Do(Returns200, Returns500))
//
//		func Returns500(b *RouteBuilder) {
//			b.Returns(500, "Internal Server Error", restful.ServiceError{})
//		}
func (b *RouteBuilder) Do(oneArgBlocks ...func(*RouteBuilder)) *RouteBuilder {
	for _, each := range oneArgBlocks {
		each(b)
	}
	return b
}

// To bind the route to a function.
// If this route is matched with the incoming Http Request then call this function with the *Request,*Response pair. Required.
func (b *RouteBuilder) To(function RouteFunction) *RouteBuilder {
	b.function = function
	return b
}

// Method specifies what HTTP method to match. Required.
func (b *RouteBuilder) Method(method string) *RouteBuilder {
	b.httpMethod = method
	return b
}

// Produces specifies what MIME types can be produced ; the matched one will appear in the Content-Type Http header.
func (b *RouteBuilder) Produces(mimeTypes ...string) *RouteBuilder {
	b.produces = mimeTypes
	return b
}

// Consumes specifies what MIME types can be consumes ; the Accept Http header must matched any of these
func (b *RouteBuilder) Consumes(mimeTypes ...string) *RouteBuilder {
	b.consumes = mimeTypes
	return b
}

// Path specifies the relative (w.r.t WebService root path) URL path to match. Default is "/".
func (b *RouteBuilder) Path(subPath string) *RouteBuilder {
	b.currentPath = subPath
	return b
}

// Doc tells what this route is all about. Optional.
func (b *RouteBuilder) Doc(documentation string) *RouteBuilder {
	b.doc = documentation
	return b
}

// A verbose explanation of the operation behavior. Optional.
func (b *RouteBuilder) Notes(notes string) *RouteBuilder {
	b.notes = notes
	return b
}

// Reads tells what resource type will be read from the request payload. Optional.
// A parameter of type "body" is added ,required is set to true and the dataType is set to the qualified name of the sample's type.
func (b *RouteBuilder) Reads(sample interface{}) *RouteBuilder {
	b.readSample = sample
	typeAsName := reflect.TypeOf(sample).String()
	bodyParameter := &Parameter{&ParameterData{Name: "body"}}
	bodyParameter.beBody()
	bodyParameter.Required(true)
	bodyParameter.DataType(typeAsName)
	b.Param(bodyParameter)
	return b
}

// ParameterNamed returns a Parameter already known to the RouteBuilder. Returns nil if not.
// Use this to modify or extend information for the Parameter (through its Data()).
func (b RouteBuilder) ParameterNamed(name string) (p *Parameter) {
	for _, each := range b.parameters {
		if each.Data().Name == name {
			return each
		}
	}
	return p
}

// Writes tells what resource type will be written as the response payload. Optional.
func (b *RouteBuilder) Writes(sample interface{}) *RouteBuilder {
	b.writeSample = sample
	return b
}

// Param allows you to document the parameters of the Route. It adds a new Parameter (does not check for duplicates).
func (b *RouteBuilder) Param(parameter *Parameter) *RouteBuilder {
	if b.parameters == nil {
		b.parameters = []*Parameter{}
	}
	b.parameters = append(b.parameters, parameter)
	return b
}

// Operation allows you to document what the acutal method/function call is of the Route.
// Unless called, the operation name is derived from the RouteFunction set using To(..).
func (b *RouteBuilder) Operation(name string) *RouteBuilder {
	b.operation = name
	return b
}

// ReturnsError is deprecated, use Returns instead.
func (b *RouteBuilder) ReturnsError(code int, message string, model interface{}) *RouteBuilder {
	log.Print("ReturnsError is deprecated, use Returns instead.")
	return b.Returns(code, message, model)
}

// Returns allows you to document what responses (errors or regular) can be expected.
// The model parameter is optional ; either pass a struct instance or use nil if not applicable.
func (b *RouteBuilder) Returns(code int, message string, model interface{}) *RouteBuilder {
	err := ResponseError{
		Code:    code,
		Message: message,
		Model:   model,
	}
	// lazy init because there is no NewRouteBuilder (yet)
	if b.errorMap == nil {
		b.errorMap = map[int]ResponseError{}
	}
	b.errorMap[code] = err
	return b
}

type ResponseError struct {
	Code    int
	Message string
	Model   interface{}
}

func (b *RouteBuilder) servicePath(path string) *RouteBuilder {
	b.rootPath = path
	return b
}

// Filter appends a FilterFunction to the end of filters for this Route to build.
func (b *RouteBuilder) Filter(filter FilterFunction) *RouteBuilder {
	b.filters = append(b.filters, filter)
	return b
}

// If no specific Route path then set to rootPath
// If no specific Produces then set to rootProduces
// If no specific Consumes then set to rootConsumes
func (b *RouteBuilder) copyDefaults(rootProduces, rootConsumes []string) {
	if len(b.produces) == 0 {
		b.produces = rootProduces
	}
	if len(b.consumes) == 0 {
		b.consumes = rootConsumes
	}
}

// Build creates a new Route using the specification details collected by the RouteBuilder
func (b *RouteBuilder) Build() Route {
	pathExpr, err := newPathExpression(b.currentPath)
	if err != nil {
		log.Printf("[restful] Invalid path:%s because:%v", b.currentPath, err)
		os.Exit(1)
	}
	if b.function == nil {
		log.Printf("[restful] No function specified for route:" + b.currentPath)
		os.Exit(1)
	}
	operationName := b.operation
	if len(operationName) == 0 && b.function != nil {
		// extract from definition
		operationName = nameOfFunction(b.function)
	}
	route := Route{
		Method:         b.httpMethod,
		Path:           concatPath(b.rootPath, b.currentPath),
		Produces:       b.produces,
		Consumes:       b.consumes,
		Function:       b.function,
		Filters:        b.filters,
		relativePath:   b.currentPath,
		pathExpr:       pathExpr,
		Doc:            b.doc,
		Notes:          b.notes,
		Operation:      operationName,
		ParameterDocs:  b.parameters,
		ResponseErrors: b.errorMap,
		ReadSample:     b.readSample,
		WriteSample:    b.writeSample}
	route.postBuild()
	return route
}

func concatPath(path1, path2 string) string {
	return strings.TrimRight(path1, "/") + "/" + strings.TrimLeft(path2, "/")
}

// nameOfFunction returns the short name of the function f for documentation.
// It uses a runtime feature for debugging ; its value may change for later Go versions.
func nameOfFunction(f interface{}) string {
	fun := runtime.FuncForPC(reflect.ValueOf(f).Pointer())
	tokenized := strings.Split(fun.Name(), ".")
	last := tokenized[len(tokenized)-1]
	last = strings.TrimSuffix(last, ")·fm") // < Go 1.5
	last = strings.TrimSuffix(last, ")-fm") // Go 1.5
	last = strings.TrimSuffix(last, "·fm")  // < Go 1.5
	last = strings.TrimSuffix(last, "-fm")  // Go 1.5
	return last
}
