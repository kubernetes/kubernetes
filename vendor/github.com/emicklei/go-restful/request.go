package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"compress/zlib"
	"io/ioutil"
	"net/http"
)

var defaultRequestContentType string

var doCacheReadEntityBytes = true

// Request is a wrapper for a http Request that provides convenience methods
type Request struct {
	Request           *http.Request
	bodyContent       *[]byte // to cache the request body for multiple reads of ReadEntity
	pathParameters    map[string]string
	attributes        map[string]interface{} // for storing request-scoped values
	selectedRoutePath string                 // root path + route path that matched the request, e.g. /meetings/{id}/attendees
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

// SetCacheReadEntity controls whether the response data ([]byte) is cached such that ReadEntity is repeatable.
// Default is true (due to backwardcompatibility). For better performance, you should set it to false if you don't need it.
func SetCacheReadEntity(doCache bool) {
	doCacheReadEntityBytes = doCache
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

	// OLD feature, cache the body for reads
	if doCacheReadEntityBytes {
		if r.bodyContent == nil {
			data, err := ioutil.ReadAll(r.Request.Body)
			if err != nil {
				return err
			}
			r.bodyContent = &data
		}
		r.Request.Body = ioutil.NopCloser(bytes.NewReader(*r.bodyContent))
	}

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

	// lookup the EntityReader
	entityReader, ok := entityAccessRegistry.AccessorAt(contentType)
	if !ok {
		return NewError(http.StatusBadRequest, "Unable to unmarshal content of type:"+contentType)
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
func (r Request) SelectedRoutePath() string {
	return r.selectedRoutePath
}
