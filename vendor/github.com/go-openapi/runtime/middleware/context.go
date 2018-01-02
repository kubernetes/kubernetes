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

package middleware

import (
	"net/http"
	"strings"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/errors"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/runtime/middleware/untyped"
	"github.com/go-openapi/runtime/security"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/gorilla/context"
)

// A Builder can create middlewares
type Builder func(http.Handler) http.Handler

// PassthroughBuilder returns the handler, aka the builder identity function
func PassthroughBuilder(handler http.Handler) http.Handler { return handler }

// RequestBinder is an interface for types to implement
// when they want to be able to bind from a request
type RequestBinder interface {
	BindRequest(*http.Request, *MatchedRoute) error
}

// Responder is an interface for types to implement
// when they want to be considered for writing HTTP responses
type Responder interface {
	WriteResponse(http.ResponseWriter, runtime.Producer)
}

// ResponderFunc wraps a func as a Responder interface
type ResponderFunc func(http.ResponseWriter, runtime.Producer)

// WriteResponse writes to the response
func (fn ResponderFunc) WriteResponse(rw http.ResponseWriter, pr runtime.Producer) {
	fn(rw, pr)
}

// Context is a type safe wrapper around an untyped request context
// used throughout to store request context with the gorilla context module
type Context struct {
	spec     *loads.Document
	analyzer *analysis.Spec
	api      RoutableAPI
	router   Router
	formats  strfmt.Registry
}

type routableUntypedAPI struct {
	api             *untyped.API
	handlers        map[string]map[string]http.Handler
	defaultConsumes string
	defaultProduces string
}

func newRoutableUntypedAPI(spec *loads.Document, api *untyped.API, context *Context) *routableUntypedAPI {
	var handlers map[string]map[string]http.Handler
	if spec == nil || api == nil {
		return nil
	}
	analyzer := analysis.New(spec.Spec())
	for method, hls := range analyzer.Operations() {
		um := strings.ToUpper(method)
		for path, op := range hls {
			schemes := analyzer.SecurityDefinitionsFor(op)

			if oh, ok := api.OperationHandlerFor(method, path); ok {
				if handlers == nil {
					handlers = make(map[string]map[string]http.Handler)
				}
				if b, ok := handlers[um]; !ok || b == nil {
					handlers[um] = make(map[string]http.Handler)
				}

				handlers[um][path] = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					// lookup route info in the context
					route, _ := context.RouteInfo(r)

					// bind and validate the request using reflection
					bound, validation := context.BindAndValidate(r, route)
					if validation != nil {
						context.Respond(w, r, route.Produces, route, validation)
						return
					}

					// actually handle the request
					result, err := oh.Handle(bound)
					if err != nil {
						// respond with failure
						context.Respond(w, r, route.Produces, route, err)
						return
					}

					// respond with success
					context.Respond(w, r, route.Produces, route, result)
				})

				if len(schemes) > 0 {
					handlers[um][path] = newSecureAPI(context, handlers[um][path])
				}
			}
		}
	}

	return &routableUntypedAPI{
		api:             api,
		handlers:        handlers,
		defaultProduces: api.DefaultProduces,
		defaultConsumes: api.DefaultConsumes,
	}
}

func (r *routableUntypedAPI) HandlerFor(method, path string) (http.Handler, bool) {
	paths, ok := r.handlers[strings.ToUpper(method)]
	if !ok {
		return nil, false
	}
	handler, ok := paths[path]
	return handler, ok
}
func (r *routableUntypedAPI) ServeErrorFor(operationID string) func(http.ResponseWriter, *http.Request, error) {
	return r.api.ServeError
}
func (r *routableUntypedAPI) ConsumersFor(mediaTypes []string) map[string]runtime.Consumer {
	return r.api.ConsumersFor(mediaTypes)
}
func (r *routableUntypedAPI) ProducersFor(mediaTypes []string) map[string]runtime.Producer {
	return r.api.ProducersFor(mediaTypes)
}
func (r *routableUntypedAPI) AuthenticatorsFor(schemes map[string]spec.SecurityScheme) map[string]runtime.Authenticator {
	return r.api.AuthenticatorsFor(schemes)
}
func (r *routableUntypedAPI) Formats() strfmt.Registry {
	return r.api.Formats()
}

func (r *routableUntypedAPI) DefaultProduces() string {
	return r.defaultProduces
}

func (r *routableUntypedAPI) DefaultConsumes() string {
	return r.defaultConsumes
}

// NewRoutableContext creates a new context for a routable API
func NewRoutableContext(spec *loads.Document, routableAPI RoutableAPI, routes Router) *Context {
	var an *analysis.Spec
	if spec != nil {
		an = analysis.New(spec.Spec())
	}
	ctx := &Context{spec: spec, api: routableAPI, analyzer: an}
	return ctx
}

// NewContext creates a new context wrapper
func NewContext(spec *loads.Document, api *untyped.API, routes Router) *Context {
	var an *analysis.Spec
	if spec != nil {
		an = analysis.New(spec.Spec())
	}
	ctx := &Context{spec: spec, analyzer: an}
	ctx.api = newRoutableUntypedAPI(spec, api, ctx)
	return ctx
}

// Serve serves the specified spec with the specified api registrations as a http.Handler
func Serve(spec *loads.Document, api *untyped.API) http.Handler {
	return ServeWithBuilder(spec, api, PassthroughBuilder)
}

// ServeWithBuilder serves the specified spec with the specified api registrations as a http.Handler that is decorated
// by the Builder
func ServeWithBuilder(spec *loads.Document, api *untyped.API, builder Builder) http.Handler {
	context := NewContext(spec, api, nil)
	return context.APIHandler(builder)
}

type contextKey int8

const (
	_ contextKey = iota
	ctxContentType
	ctxResponseFormat
	ctxMatchedRoute
	ctxAllowedMethods
	ctxBoundParams
	ctxSecurityPrincipal
	ctxSecurityScopes

	ctxConsumer
)

type contentTypeValue struct {
	MediaType string
	Charset   string
}

// BasePath returns the base path for this API
func (c *Context) BasePath() string {
	return c.spec.BasePath()
}

// RequiredProduces returns the accepted content types for responses
func (c *Context) RequiredProduces() []string {
	return c.analyzer.RequiredProduces()
}

// BindValidRequest binds a params object to a request but only when the request is valid
// if the request is not valid an error will be returned
func (c *Context) BindValidRequest(request *http.Request, route *MatchedRoute, binder RequestBinder) error {
	var res []error

	requestContentType := "*/*"
	// check and validate content type, select consumer
	if runtime.HasBody(request) {
		ct, _, err := runtime.ContentType(request.Header)
		if err != nil {
			res = append(res, err)
		} else {
			if err := validateContentType(route.Consumes, ct); err != nil {
				res = append(res, err)
			}
			if len(res) == 0 {
				cons, ok := route.Consumers[ct]
				if !ok {
					res = append(res, errors.New(500, "no consumer registered for %s", ct))
				} else {
					route.Consumer = cons
					requestContentType = ct
				}
			}
		}
	}

	// check and validate the response format
	if len(res) == 0 && runtime.HasBody(request) {
		if str := NegotiateContentType(request, route.Produces, requestContentType); str == "" {
			res = append(res, errors.InvalidResponseFormat(request.Header.Get(runtime.HeaderAccept), route.Produces))
		}
	}

	// now bind the request with the provided binder
	// it's assumed the binder will also validate the request and return an error if the
	// request is invalid
	if binder != nil && len(res) == 0 {
		if err := binder.BindRequest(request, route); err != nil {
			return err
		}
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}

// ContentType gets the parsed value of a content type
func (c *Context) ContentType(request *http.Request) (string, string, error) {
	if v, ok := context.GetOk(request, ctxContentType); ok {
		if val, ok := v.(*contentTypeValue); ok {
			return val.MediaType, val.Charset, nil
		}
	}

	mt, cs, err := runtime.ContentType(request.Header)
	if err != nil {
		return "", "", err
	}
	context.Set(request, ctxContentType, &contentTypeValue{mt, cs})
	return mt, cs, nil
}

// LookupRoute looks a route up and returns true when it is found
func (c *Context) LookupRoute(request *http.Request) (*MatchedRoute, bool) {
	if route, ok := c.router.Lookup(request.Method, request.URL.Path); ok {
		return route, ok
	}
	return nil, false
}

// RouteInfo tries to match a route for this request
func (c *Context) RouteInfo(request *http.Request) (*MatchedRoute, bool) {
	if v, ok := context.GetOk(request, ctxMatchedRoute); ok {
		if val, ok := v.(*MatchedRoute); ok {
			return val, ok
		}
	}

	if route, ok := c.LookupRoute(request); ok {
		context.Set(request, ctxMatchedRoute, route)
		return route, ok
	}

	return nil, false
}

// ResponseFormat negotiates the response content type
func (c *Context) ResponseFormat(r *http.Request, offers []string) string {
	if v, ok := context.GetOk(r, ctxResponseFormat); ok {
		if val, ok := v.(string); ok {
			return val
		}
	}

	format := NegotiateContentType(r, offers, "")
	if format != "" {
		context.Set(r, ctxResponseFormat, format)
	}
	return format
}

// AllowedMethods gets the allowed methods for the path of this request
func (c *Context) AllowedMethods(request *http.Request) []string {
	return c.router.OtherMethods(request.Method, request.URL.Path)
}

// Authorize authorizes the request
func (c *Context) Authorize(request *http.Request, route *MatchedRoute) (interface{}, error) {
	if len(route.Authenticators) == 0 {
		return nil, nil
	}
	if v, ok := context.GetOk(request, ctxSecurityPrincipal); ok {
		return v, nil
	}

	for scheme, authenticator := range route.Authenticators {
		applies, usr, err := authenticator.Authenticate(&security.ScopedAuthRequest{
			Request:        request,
			RequiredScopes: route.Scopes[scheme],
		})
		if !applies || err != nil || usr == nil {
			continue
		}
		context.Set(request, ctxSecurityPrincipal, usr)
		context.Set(request, ctxSecurityScopes, route.Scopes[scheme])
		return usr, nil
	}

	return nil, errors.Unauthenticated("invalid credentials")
}

// BindAndValidate binds and validates the request
func (c *Context) BindAndValidate(request *http.Request, matched *MatchedRoute) (interface{}, error) {
	if v, ok := context.GetOk(request, ctxBoundParams); ok {
		if val, ok := v.(*validation); ok {
			if len(val.result) > 0 {
				return val.bound, errors.CompositeValidationError(val.result...)
			}
			return val.bound, nil
		}
	}
	result := validateRequest(c, request, matched)
	if result != nil {
		context.Set(request, ctxBoundParams, result)
	}
	if len(result.result) > 0 {
		return result.bound, errors.CompositeValidationError(result.result...)
	}
	return result.bound, nil
}

// NotFound the default not found responder for when no route has been matched yet
func (c *Context) NotFound(rw http.ResponseWriter, r *http.Request) {
	c.Respond(rw, r, []string{c.api.DefaultProduces()}, nil, errors.NotFound("not found"))
}

// Respond renders the response after doing some content negotiation
func (c *Context) Respond(rw http.ResponseWriter, r *http.Request, produces []string, route *MatchedRoute, data interface{}) {
	offers := []string{}
	for _, mt := range produces {
		if mt != c.api.DefaultProduces() {
			offers = append(offers, mt)
		}
	}
	// the default producer is last so more specific producers take precedence
	offers = append(offers, c.api.DefaultProduces())

	format := c.ResponseFormat(r, offers)
	rw.Header().Set(runtime.HeaderContentType, format)

	if resp, ok := data.(Responder); ok {
		producers := route.Producers
		prod, ok := producers[format]
		if !ok {
			prods := c.api.ProducersFor([]string{c.api.DefaultProduces()})
			pr, ok := prods[c.api.DefaultProduces()]
			if !ok {
				panic(errors.New(http.StatusInternalServerError, "can't find a producer for "+format))
			}
			prod = pr
		}
		resp.WriteResponse(rw, prod)
		return
	}

	if err, ok := data.(error); ok {
		if format == "" {
			rw.Header().Set(runtime.HeaderContentType, runtime.JSONMime)
		}
		if route == nil || route.Operation == nil {
			c.api.ServeErrorFor("")(rw, r, err)
			return
		}
		c.api.ServeErrorFor(route.Operation.ID)(rw, r, err)
		return
	}

	if route == nil || route.Operation == nil {
		rw.WriteHeader(200)
		if r.Method == "HEAD" {
			return
		}
		producers := c.api.ProducersFor(offers)
		prod, ok := producers[format]
		if !ok {
			panic(errors.New(http.StatusInternalServerError, "can't find a producer for "+format))
		}
		if err := prod.Produce(rw, data); err != nil {
			panic(err) // let the recovery middleware deal with this
		}
		return
	}

	if _, code, ok := route.Operation.SuccessResponse(); ok {
		rw.WriteHeader(code)
		if code == 204 || r.Method == "HEAD" {
			return
		}

		producers := route.Producers
		prod, ok := producers[format]
		if !ok {
			if !ok {
				prods := c.api.ProducersFor([]string{c.api.DefaultProduces()})
				pr, ok := prods[c.api.DefaultProduces()]
				if !ok {
					panic(errors.New(http.StatusInternalServerError, "can't find a producer for "+format))
				}
				prod = pr
			}
		}
		if err := prod.Produce(rw, data); err != nil {
			panic(err) // let the recovery middleware deal with this
		}
		return
	}

	c.api.ServeErrorFor(route.Operation.ID)(rw, r, errors.New(http.StatusInternalServerError, "can't produce response"))
}

// APIHandler returns a handler to serve
func (c *Context) APIHandler(builder Builder) http.Handler {
	b := builder
	if b == nil {
		b = PassthroughBuilder
	}
	return specMiddleware(c, newRouter(c, b(newOperationExecutor(c))))
}
