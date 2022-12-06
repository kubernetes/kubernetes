package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"compress/zlib"
	"net/http"
)

var defaultRequestContentType string

// Request is a wrapper for a http Request that provides convenience methods
type Request struct {
	Request        *http.Request
	pathParameters map[string]string
	attributes     map[string]interface{} // for storing request-scoped values
	selectedRoute  *Route                 // is nil when no route was matched
}

func NewRequest(httpRequest *http.Request) *Request {
	return &Request{
		Request:        httpRequest,
		pathParameters: map[string]string{},
		attributes:     map[string]interface{}{},
	} // empty parameters, attributes
}

// If ContentType is missing or */* is given then fall back to this type, otherwise
// a "Unable to unmarshal content of type:" response is returned.
// Valid values are restful.MIME_JSON and restful.MIME_XML
// Example:
// 	restful.DefaultRequestContentType(restful.MIME_JSON)
func DefaultRequestContentType(mime string) {
	defaultRequestContentType = mime
}

// PathParameter accesses the Path parameter value by its name
func (r *Request) PathParameter(name string) string {
	return r.pathParameters[name]
}

// PathParameters accesses the Path parameter values
func (r *Request) PathParameters() map[string]string {
	return r.pathParameters
}

// QueryParameter returns the (first) Query parameter value by its name
func (r *Request) QueryParameter(name string) string {
	return r.Request.FormValue(name)
}

// QueryParameters returns the all the query parameters values by name
func (r *Request) QueryParameters(name string) []string {
	return r.Request.URL.Query()[name]
}

// BodyParameter parses the body of the request (once for typically a POST or a PUT) and returns the value of the given name or an error.
func (r *Request) BodyParameter(name string) (string, error) {
	err := r.Request.ParseForm()
	if err != nil {
		return "", err
	}
	return r.Request.PostFormValue(name), nil
}

// HeaderParameter returns the HTTP Header value of a Header name or empty if missing
func (r *Request) HeaderParameter(name string) string {
	return r.Request.Header.Get(name)
}

// ReadEntity checks the Accept header and reads the content into the entityPointer.
func (r *Request) ReadEntity(entityPointer interface{}) (err error) {
	contentType := r.Request.Header.Get(HEADER_ContentType)
	contentEncoding := r.Request.Header.Get(HEADER_ContentEncoding)

	// check if the request body needs decompression
	if ENCODING_GZIP == contentEncoding {
		gzipReader := currentCompressorProvider.AcquireGzipReader()
		defer currentCompressorProvider.ReleaseGzipReader(gzipReader)
		gzipReader.Reset(r.Request.Body)
		r.Request.Body = gzipReader
	} else if ENCODING_DEFLATE == contentEncoding {
		zlibReader, err := zlib.NewReader(r.Request.Body)
		if err != nil {
			return err
		}
		r.Request.Body = zlibReader
	}

	// lookup the EntityReader, use defaultRequestContentType if needed and provided
	entityReader, ok := entityAccessRegistry.accessorAt(contentType)
	if !ok {
		if len(defaultRequestContentType) != 0 {
			entityReader, ok = entityAccessRegistry.accessorAt(defaultRequestContentType)
		}
		if !ok {
			return NewError(http.StatusBadRequest, "Unable to unmarshal content of type:"+contentType)
		}
	}
	return entityReader.Read(r, entityPointer)
}

// SetAttribute adds or replaces the attribute with the given value.
func (r *Request) SetAttribute(name string, value interface{}) {
	r.attributes[name] = value
}

// Attribute returns the value associated to the given name. Returns nil if absent.
func (r Request) Attribute(name string) interface{} {
	return r.attributes[name]
}

// SelectedRoutePath root path + route path that matched the request, e.g. /meetings/{id}/attendees
// If no route was matched then return an empty string.
func (r Request) SelectedRoutePath() string {
	if r.selectedRoute == nil {
		return ""
	}
	// skip creating an accessor
	return r.selectedRoute.Path
}

// SelectedRoute returns a reader to access the selected Route by the container
// Returns nil if no route was matched.
func (r Request) SelectedRoute() RouteReader {
	if r.selectedRoute == nil {
		return nil
	}
	return routeAccessor{route: r.selectedRoute}
}
