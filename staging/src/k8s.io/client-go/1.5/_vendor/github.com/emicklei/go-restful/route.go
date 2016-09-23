package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"net/http"
	"strings"
)

// RouteFunction declares the signature of a function that can be bound to a Route.
type RouteFunction func(*Request, *Response)

// Route binds a HTTP Method,Path,Consumes combination to a RouteFunction.
type Route struct {
	Method   string
	Produces []string
	Consumes []string
	Path     string // webservice root path + described path
	Function RouteFunction
	Filters  []FilterFunction

	// cached values for dispatching
	relativePath string
	pathParts    []string
	pathExpr     *pathExpression // cached compilation of relativePath as RegExp

	// documentation
	Doc                     string
	Notes                   string
	Operation               string
	ParameterDocs           []*Parameter
	ResponseErrors          map[int]ResponseError
	ReadSample, WriteSample interface{} // structs that model an example request or response payload
}

// Initialize for Route
func (r *Route) postBuild() {
	r.pathParts = tokenizePath(r.Path)
}

// Create Request and Response from their http versions
func (r *Route) wrapRequestResponse(httpWriter http.ResponseWriter, httpRequest *http.Request) (*Request, *Response) {
	params := r.extractParameters(httpRequest.URL.Path)
	wrappedRequest := NewRequest(httpRequest)
	wrappedRequest.pathParameters = params
	wrappedRequest.selectedRoutePath = r.Path
	wrappedResponse := NewResponse(httpWriter)
	wrappedResponse.requestAccept = httpRequest.Header.Get(HEADER_Accept)
	wrappedResponse.routeProduces = r.Produces
	return wrappedRequest, wrappedResponse
}

// dispatchWithFilters call the function after passing through its own filters
func (r *Route) dispatchWithFilters(wrappedRequest *Request, wrappedResponse *Response) {
	if len(r.Filters) > 0 {
		chain := FilterChain{Filters: r.Filters, Target: r.Function}
		chain.ProcessFilter(wrappedRequest, wrappedResponse)
	} else {
		// unfiltered
		r.Function(wrappedRequest, wrappedResponse)
	}
}

// Return whether the mimeType matches to what this Route can produce.
func (r Route) matchesAccept(mimeTypesWithQuality string) bool {
	parts := strings.Split(mimeTypesWithQuality, ",")
	for _, each := range parts {
		var withoutQuality string
		if strings.Contains(each, ";") {
			withoutQuality = strings.Split(each, ";")[0]
		} else {
			withoutQuality = each
		}
		// trim before compare
		withoutQuality = strings.Trim(withoutQuality, " ")
		if withoutQuality == "*/*" {
			return true
		}
		for _, producibleType := range r.Produces {
			if producibleType == "*/*" || producibleType == withoutQuality {
				return true
			}
		}
	}
	return false
}

// Return whether this Route can consume content with a type specified by mimeTypes (can be empty).
func (r Route) matchesContentType(mimeTypes string) bool {

	if len(r.Consumes) == 0 {
		// did not specify what it can consume ;  any media type (“*/*”) is assumed
		return true
	}

	if len(mimeTypes) == 0 {
		// idempotent methods with (most-likely or garanteed) empty content match missing Content-Type
		m := r.Method
		if m == "GET" || m == "HEAD" || m == "OPTIONS" || m == "DELETE" || m == "TRACE" {
			return true
		}
		// proceed with default
		mimeTypes = MIME_OCTET
	}

	parts := strings.Split(mimeTypes, ",")
	for _, each := range parts {
		var contentType string
		if strings.Contains(each, ";") {
			contentType = strings.Split(each, ";")[0]
		} else {
			contentType = each
		}
		// trim before compare
		contentType = strings.Trim(contentType, " ")
		for _, consumeableType := range r.Consumes {
			if consumeableType == "*/*" || consumeableType == contentType {
				return true
			}
		}
	}
	return false
}

// Extract the parameters from the request url path
func (r Route) extractParameters(urlPath string) map[string]string {
	urlParts := tokenizePath(urlPath)
	pathParameters := map[string]string{}
	for i, key := range r.pathParts {
		var value string
		if i >= len(urlParts) {
			value = ""
		} else {
			value = urlParts[i]
		}
		if strings.HasPrefix(key, "{") { // path-parameter
			if colon := strings.Index(key, ":"); colon != -1 {
				// extract by regex
				regPart := key[colon+1 : len(key)-1]
				keyPart := key[1:colon]
				if regPart == "*" {
					pathParameters[keyPart] = untokenizePath(i, urlParts)
					break
				} else {
					pathParameters[keyPart] = value
				}
			} else {
				// without enclosing {}
				pathParameters[key[1:len(key)-1]] = value
			}
		}
	}
	return pathParameters
}

// Untokenize back into an URL path using the slash separator
func untokenizePath(offset int, parts []string) string {
	var buffer bytes.Buffer
	for p := offset; p < len(parts); p++ {
		buffer.WriteString(parts[p])
		// do not end
		if p < len(parts)-1 {
			buffer.WriteString("/")
		}
	}
	return buffer.String()
}

// Tokenize an URL path using the slash separator ; the result does not have empty tokens
func tokenizePath(path string) []string {
	if "/" == path {
		return []string{}
	}
	return strings.Split(strings.Trim(path, "/"), "/")
}

// for debugging
func (r Route) String() string {
	return r.Method + " " + r.Path
}
