package restful

import (
	"errors"
	"os"
	"reflect"
	"sync"

	"github.com/emicklei/go-restful/v3/log"
)

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

// WebService holds a collection of Route values that bind a Http Method + URL Path to a function.
type WebService struct {
	rootPath       string
	pathExpr       *pathExpression // cached compilation of rootPath as RegExp
	routes         []Route
	produces       []string
	consumes       []string
	pathParameters []*Parameter
	filters        []FilterFunction
	documentation  string
	apiVersion     string

	typeNameHandleFunc TypeNameHandleFunction

	dynamicRoutes bool

	// protects 'routes' if dynamic routes are enabled
	routesLock sync.RWMutex
}

func (w *WebService) SetDynamicRoutes(enable bool) {
	w.dynamicRoutes = enable
}

// TypeNameHandleFunction declares functions that can handle translating the name of a sample object
// into the restful documentation for the service.
type TypeNameHandleFunction func(sample interface{}) string

// TypeNameHandler sets the function that will convert types to strings in the parameter
// and model definitions. If not set, the web service will invoke
// reflect.TypeOf(object).String().
func (w *WebService) TypeNameHandler(handler TypeNameHandleFunction) *WebService {
	w.typeNameHandleFunc = handler
	return w
}

// reflectTypeName is the default TypeNameHandleFunction and for a given object
// returns the name that Go identifies it with (e.g. "string" or "v1.Object") via
// the reflection API.
func reflectTypeName(sample interface{}) string {
	return reflect.TypeOf(sample).String()
}

// compilePathExpression ensures that the path is compiled into a RegEx for those routers that need it.
func (w *WebService) compilePathExpression() {
	compiled, err := newPathExpression(w.rootPath)
	if err != nil {
		log.Printf("invalid path:%s because:%v", w.rootPath, err)
		os.Exit(1)
	}
	w.pathExpr = compiled
}

// ApiVersion sets the API version for documentation purposes.
func (w *WebService) ApiVersion(apiVersion string) *WebService {
	w.apiVersion = apiVersion
	return w
}

// Version returns the API version for documentation purposes.
func (w *WebService) Version() string { return w.apiVersion }

// Path specifies the root URL template path of the WebService.
// All Routes will be relative to this path.
func (w *WebService) Path(root string) *WebService {
	w.rootPath = root
	if len(w.rootPath) == 0 {
		w.rootPath = "/"
	}
	w.compilePathExpression()
	return w
}

// Param adds a PathParameter to document parameters used in the root path.
func (w *WebService) Param(parameter *Parameter) *WebService {
	if w.pathParameters == nil {
		w.pathParameters = []*Parameter{}
	}
	w.pathParameters = append(w.pathParameters, parameter)
	return w
}

// PathParameter creates a new Parameter of kind Path for documentation purposes.
// It is initialized as required with string as its DataType.
func (w *WebService) PathParameter(name, description string) *Parameter {
	return PathParameter(name, description)
}

// PathParameter creates a new Parameter of kind Path for documentation purposes.
// It is initialized as required with string as its DataType.
func PathParameter(name, description string) *Parameter {
	p := &Parameter{&ParameterData{Name: name, Description: description, Required: true, DataType: "string"}}
	p.bePath()
	return p
}

// QueryParameter creates a new Parameter of kind Query for documentation purposes.
// It is initialized as not required with string as its DataType.
func (w *WebService) QueryParameter(name, description string) *Parameter {
	return QueryParameter(name, description)
}

// QueryParameter creates a new Parameter of kind Query for documentation purposes.
// It is initialized as not required with string as its DataType.
func QueryParameter(name, description string) *Parameter {
	p := &Parameter{&ParameterData{Name: name, Description: description, Required: false, DataType: "string", CollectionFormat: CollectionFormatCSV.String()}}
	p.beQuery()
	return p
}

// BodyParameter creates a new Parameter of kind Body for documentation purposes.
// It is initialized as required without a DataType.
func (w *WebService) BodyParameter(name, description string) *Parameter {
	return BodyParameter(name, description)
}

// BodyParameter creates a new Parameter of kind Body for documentation purposes.
// It is initialized as required without a DataType.
func BodyParameter(name, description string) *Parameter {
	p := &Parameter{&ParameterData{Name: name, Description: description, Required: true}}
	p.beBody()
	return p
}

// HeaderParameter creates a new Parameter of kind (Http) Header for documentation purposes.
// It is initialized as not required with string as its DataType.
func (w *WebService) HeaderParameter(name, description string) *Parameter {
	return HeaderParameter(name, description)
}

// HeaderParameter creates a new Parameter of kind (Http) Header for documentation purposes.
// It is initialized as not required with string as its DataType.
func HeaderParameter(name, description string) *Parameter {
	p := &Parameter{&ParameterData{Name: name, Description: description, Required: false, DataType: "string"}}
	p.beHeader()
	return p
}

// FormParameter creates a new Parameter of kind Form (using application/x-www-form-urlencoded) for documentation purposes.
// It is initialized as required with string as its DataType.
func (w *WebService) FormParameter(name, description string) *Parameter {
	return FormParameter(name, description)
}

// FormParameter creates a new Parameter of kind Form (using application/x-www-form-urlencoded) for documentation purposes.
// It is initialized as required with string as its DataType.
func FormParameter(name, description string) *Parameter {
	p := &Parameter{&ParameterData{Name: name, Description: description, Required: false, DataType: "string"}}
	p.beForm()
	return p
}

// Route creates a new Route using the RouteBuilder and add to the ordered list of Routes.
func (w *WebService) Route(builder *RouteBuilder) *WebService {
	w.routesLock.Lock()
	defer w.routesLock.Unlock()
	builder.copyDefaults(w.produces, w.consumes)
	w.routes = append(w.routes, builder.Build())
	return w
}

// RemoveRoute removes the specified route, looks for something that matches 'path' and 'method'
func (w *WebService) RemoveRoute(path, method string) error {
    if !w.dynamicRoutes {
        return errors.New("dynamic routes are not enabled.")
    }
    w.routesLock.Lock()
    defer w.routesLock.Unlock()
    newRoutes := []Route{}
    for _, route := range w.routes {
        if route.Method == method && route.Path == path {
            continue
        }
        newRoutes = append(newRoutes, route)
    }
    w.routes = newRoutes
    return nil
}

// Method creates a new RouteBuilder and initialize its http method
func (w *WebService) Method(httpMethod string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method(httpMethod)
}

// Produces specifies that this WebService can produce one or more MIME types.
// Http requests must have one of these values set for the Accept header.
func (w *WebService) Produces(contentTypes ...string) *WebService {
	w.produces = contentTypes
	return w
}

// Consumes specifies that this WebService can consume one or more MIME types.
// Http requests must have one of these values set for the Content-Type header.
func (w *WebService) Consumes(accepts ...string) *WebService {
	w.consumes = accepts
	return w
}

// Routes returns the Routes associated with this WebService
func (w *WebService) Routes() []Route {
	if !w.dynamicRoutes {
		return w.routes
	}
	// Make a copy of the array to prevent concurrency problems
	w.routesLock.RLock()
	defer w.routesLock.RUnlock()
	result := make([]Route, len(w.routes))
	for ix := range w.routes {
		result[ix] = w.routes[ix]
	}
	return result
}

// RootPath returns the RootPath associated with this WebService. Default "/"
func (w *WebService) RootPath() string {
	return w.rootPath
}

// PathParameters return the path parameter names for (shared among its Routes)
func (w *WebService) PathParameters() []*Parameter {
	return w.pathParameters
}

// Filter adds a filter function to the chain of filters applicable to all its Routes
func (w *WebService) Filter(filter FilterFunction) *WebService {
	w.filters = append(w.filters, filter)
	return w
}

// Doc is used to set the documentation of this service.
func (w *WebService) Doc(plainText string) *WebService {
	w.documentation = plainText
	return w
}

// Documentation returns it.
func (w *WebService) Documentation() string {
	return w.documentation
}

/*
	Convenience methods
*/

// HEAD is a shortcut for .Method("HEAD").Path(subPath)
func (w *WebService) HEAD(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("HEAD").Path(subPath)
}

// GET is a shortcut for .Method("GET").Path(subPath)
func (w *WebService) GET(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("GET").Path(subPath)
}

// POST is a shortcut for .Method("POST").Path(subPath)
func (w *WebService) POST(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("POST").Path(subPath)
}

// PUT is a shortcut for .Method("PUT").Path(subPath)
func (w *WebService) PUT(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("PUT").Path(subPath)
}

// PATCH is a shortcut for .Method("PATCH").Path(subPath)
func (w *WebService) PATCH(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("PATCH").Path(subPath)
}

// DELETE is a shortcut for .Method("DELETE").Path(subPath)
func (w *WebService) DELETE(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("DELETE").Path(subPath)
}

// OPTIONS is a shortcut for .Method("OPTIONS").Path(subPath)
func (w *WebService) OPTIONS(subPath string) *RouteBuilder {
	return new(RouteBuilder).typeNameHandler(w.typeNameHandleFunc).servicePath(w.rootPath).Method("OPTIONS").Path(subPath)
}
