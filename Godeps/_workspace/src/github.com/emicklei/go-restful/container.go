package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"

	"github.com/emicklei/go-restful/log"
)

// Container holds a collection of WebServices and a http.ServeMux to dispatch http requests.
// The requests are further dispatched to routes of WebServices using a RouteSelector
type Container struct {
	webServices            []*WebService
	ServeMux               *http.ServeMux
	isRegisteredOnRoot     bool
	containerFilters       []FilterFunction
	doNotRecover           bool // default is false
	recoverHandleFunc      RecoverHandleFunction
	serviceErrorHandleFunc ServiceErrorHandleFunction
	router                 RouteSelector // default is a RouterJSR311, CurlyRouter is the faster alternative
	contentEncodingEnabled bool          // default is false
}

// NewContainer creates a new Container using a new ServeMux and default router (RouterJSR311)
func NewContainer() *Container {
	return &Container{
		webServices:            []*WebService{},
		ServeMux:               http.NewServeMux(),
		isRegisteredOnRoot:     false,
		containerFilters:       []FilterFunction{},
		doNotRecover:           false,
		recoverHandleFunc:      logStackOnRecover,
		serviceErrorHandleFunc: writeServiceError,
		router:                 RouterJSR311{},
		contentEncodingEnabled: false}
}

// RecoverHandleFunction declares functions that can be used to handle a panic situation.
// The first argument is what recover() returns. The second must be used to communicate an error response.
type RecoverHandleFunction func(interface{}, http.ResponseWriter)

// RecoverHandler changes the default function (logStackOnRecover) to be called
// when a panic is detected. DoNotRecover must be have its default value (=false).
func (c *Container) RecoverHandler(handler RecoverHandleFunction) {
	c.recoverHandleFunc = handler
}

// ServiceErrorHandleFunction declares functions that can be used to handle a service error situation.
// The first argument is the service error, the second is the request that resulted in the error and
// the third must be used to communicate an error response.
type ServiceErrorHandleFunction func(ServiceError, *Request, *Response)

// ServiceErrorHandler changes the default function (writeServiceError) to be called
// when a ServiceError is detected.
func (c *Container) ServiceErrorHandler(handler ServiceErrorHandleFunction) {
	c.serviceErrorHandleFunc = handler
}

// DoNotRecover controls whether panics will be caught to return HTTP 500.
// If set to true, Route functions are responsible for handling any error situation.
// Default value is false = recover from panics. This has performance implications.
func (c *Container) DoNotRecover(doNot bool) {
	c.doNotRecover = doNot
}

// Router changes the default Router (currently RouterJSR311)
func (c *Container) Router(aRouter RouteSelector) {
	c.router = aRouter
}

// EnableContentEncoding (default=false) allows for GZIP or DEFLATE encoding of responses.
func (c *Container) EnableContentEncoding(enabled bool) {
	c.contentEncodingEnabled = enabled
}

// Add a WebService to the Container. It will detect duplicate root paths and panic in that case.
func (c *Container) Add(service *WebService) *Container {
	// If registered on root then no additional specific mapping is needed
	if !c.isRegisteredOnRoot {
		pattern := c.fixedPrefixPath(service.RootPath())
		// check if root path registration is needed
		if "/" == pattern || "" == pattern {
			c.ServeMux.HandleFunc("/", c.dispatch)
			c.isRegisteredOnRoot = true
		} else {
			// detect if registration already exists
			alreadyMapped := false
			for _, each := range c.webServices {
				if each.RootPath() == service.RootPath() {
					alreadyMapped = true
					break
				}
			}
			if !alreadyMapped {
				c.ServeMux.HandleFunc(pattern, c.dispatch)
				if !strings.HasSuffix(pattern, "/") {
					c.ServeMux.HandleFunc(pattern+"/", c.dispatch)
				}
			}
		}
	}
	// cannot have duplicate root paths
	for _, each := range c.webServices {
		if each.RootPath() == service.RootPath() {
			log.Printf("[restful] WebService with duplicate root path detected:['%v']", each)
			os.Exit(1)
		}
	}
	// if rootPath was not set then lazy initialize it
	if len(service.rootPath) == 0 {
		service.Path("/")
	}
	c.webServices = append(c.webServices, service)
	return c
}

// logStackOnRecover is the default RecoverHandleFunction and is called
// when DoNotRecover is false and the recoverHandleFunc is not set for the container.
// Default implementation logs the stacktrace and writes the stacktrace on the response.
// This may be a security issue as it exposes sourcecode information.
func logStackOnRecover(panicReason interface{}, httpWriter http.ResponseWriter) {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("[restful] recover from panic situation: - %v\r\n", panicReason))
	for i := 2; ; i += 1 {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		buffer.WriteString(fmt.Sprintf("    %s:%d\r\n", file, line))
	}
	log.Print(buffer.String())
	httpWriter.WriteHeader(http.StatusInternalServerError)
	httpWriter.Write(buffer.Bytes())
}

// writeServiceError is the default ServiceErrorHandleFunction and is called
// when a ServiceError is returned during route selection. Default implementation
// calls resp.WriteErrorString(err.Code, err.Message)
func writeServiceError(err ServiceError, req *Request, resp *Response) {
	resp.WriteErrorString(err.Code, err.Message)
}

// Dispatch the incoming Http Request to a matching WebService.
func (c *Container) dispatch(httpWriter http.ResponseWriter, httpRequest *http.Request) {
	writer := httpWriter

	// CompressingResponseWriter should be closed after all operations are done
	defer func() {
		if compressWriter, ok := writer.(*CompressingResponseWriter); ok {
			compressWriter.Close()
		}
	}()

	// Instal panic recovery unless told otherwise
	if !c.doNotRecover { // catch all for 500 response
		defer func() {
			if r := recover(); r != nil {
				c.recoverHandleFunc(r, writer)
				return
			}
		}()
	}
	// Install closing the request body (if any)
	defer func() {
		if nil != httpRequest.Body {
			httpRequest.Body.Close()
		}
	}()

	// Detect if compression is needed
	// assume without compression, test for override
	if c.contentEncodingEnabled {
		doCompress, encoding := wantsCompressedResponse(httpRequest)
		if doCompress {
			var err error
			writer, err = NewCompressingResponseWriter(httpWriter, encoding)
			if err != nil {
				log.Print("[restful] unable to install compressor: ", err)
				httpWriter.WriteHeader(http.StatusInternalServerError)
				return
			}
		}
	}
	// Find best match Route ; err is non nil if no match was found
	webService, route, err := c.router.SelectRoute(
		c.webServices,
		httpRequest)
	if err != nil {
		// a non-200 response has already been written
		// run container filters anyway ; they should not touch the response...
		chain := FilterChain{Filters: c.containerFilters, Target: func(req *Request, resp *Response) {
			switch err.(type) {
			case ServiceError:
				ser := err.(ServiceError)
				c.serviceErrorHandleFunc(ser, req, resp)
			}
			// TODO
		}}
		chain.ProcessFilter(NewRequest(httpRequest), NewResponse(writer))
		return
	}
	wrappedRequest, wrappedResponse := route.wrapRequestResponse(writer, httpRequest)
	// pass through filters (if any)
	if len(c.containerFilters)+len(webService.filters)+len(route.Filters) > 0 {
		// compose filter chain
		allFilters := []FilterFunction{}
		allFilters = append(allFilters, c.containerFilters...)
		allFilters = append(allFilters, webService.filters...)
		allFilters = append(allFilters, route.Filters...)
		chain := FilterChain{Filters: allFilters, Target: func(req *Request, resp *Response) {
			// handle request by route after passing all filters
			route.Function(wrappedRequest, wrappedResponse)
		}}
		chain.ProcessFilter(wrappedRequest, wrappedResponse)
	} else {
		// no filters, handle request by route
		route.Function(wrappedRequest, wrappedResponse)
	}
}

// fixedPrefixPath returns the fixed part of the partspec ; it may include template vars {}
func (c Container) fixedPrefixPath(pathspec string) string {
	varBegin := strings.Index(pathspec, "{")
	if -1 == varBegin {
		return pathspec
	}
	return pathspec[:varBegin]
}

// ServeHTTP implements net/http.Handler therefore a Container can be a Handler in a http.Server
func (c Container) ServeHTTP(httpwriter http.ResponseWriter, httpRequest *http.Request) {
	c.ServeMux.ServeHTTP(httpwriter, httpRequest)
}

// Handle registers the handler for the given pattern. If a handler already exists for pattern, Handle panics.
func (c Container) Handle(pattern string, handler http.Handler) {
	c.ServeMux.Handle(pattern, handler)
}

// HandleWithFilter registers the handler for the given pattern.
// Container's filter chain is applied for handler.
// If a handler already exists for pattern, HandleWithFilter panics.
func (c Container) HandleWithFilter(pattern string, handler http.Handler) {
	f := func(httpResponse http.ResponseWriter, httpRequest *http.Request) {
		if len(c.containerFilters) == 0 {
			handler.ServeHTTP(httpResponse, httpRequest)
			return
		}

		chain := FilterChain{Filters: c.containerFilters, Target: func(req *Request, resp *Response) {
			handler.ServeHTTP(httpResponse, httpRequest)
		}}
		chain.ProcessFilter(NewRequest(httpRequest), NewResponse(httpResponse))
	}

	c.Handle(pattern, http.HandlerFunc(f))
}

// Filter appends a container FilterFunction. These are called before dispatching
// a http.Request to a WebService from the container
func (c *Container) Filter(filter FilterFunction) {
	c.containerFilters = append(c.containerFilters, filter)
}

// RegisteredWebServices returns the collections of added WebServices
func (c Container) RegisteredWebServices() []*WebService {
	return c.webServices
}

// computeAllowedMethods returns a list of HTTP methods that are valid for a Request
func (c Container) computeAllowedMethods(req *Request) []string {
	// Go through all RegisteredWebServices() and all its Routes to collect the options
	methods := []string{}
	requestPath := req.Request.URL.Path
	for _, ws := range c.RegisteredWebServices() {
		matches := ws.pathExpr.Matcher.FindStringSubmatch(requestPath)
		if matches != nil {
			finalMatch := matches[len(matches)-1]
			for _, rt := range ws.Routes() {
				matches := rt.pathExpr.Matcher.FindStringSubmatch(finalMatch)
				if matches != nil {
					lastMatch := matches[len(matches)-1]
					if lastMatch == "" || lastMatch == "/" { // do not include if value is neither empty nor ‘/’.
						methods = append(methods, rt.Method)
					}
				}
			}
		}
	}
	// methods = append(methods, "OPTIONS")  not sure about this
	return methods
}

// newBasicRequestResponse creates a pair of Request,Response from its http versions.
// It is basic because no parameter or (produces) content-type information is given.
func newBasicRequestResponse(httpWriter http.ResponseWriter, httpRequest *http.Request) (*Request, *Response) {
	resp := NewResponse(httpWriter)
	resp.requestAccept = httpRequest.Header.Get(HEADER_Accept)
	return NewRequest(httpRequest), resp
}
