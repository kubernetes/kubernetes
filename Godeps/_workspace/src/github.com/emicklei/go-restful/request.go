package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"encoding/json"
	"encoding/xml"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
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

// ReadEntity checks the Accept header and reads the content into the entityPointer
// May be called multiple times in the request-response flow
func (r *Request) ReadEntity(entityPointer interface{}) (err error) {
	defer r.Request.Body.Close()
	contentType := r.Request.Header.Get(HEADER_ContentType)
	contentEncoding := r.Request.Header.Get(HEADER_ContentEncoding)
	if doCacheReadEntityBytes {
		return r.cachingReadEntity(contentType, contentEncoding, entityPointer)
	}
	// unmarshall directly from request Body
	return r.decodeEntity(r.Request.Body, contentType, contentEncoding, entityPointer)
}

func (r *Request) cachingReadEntity(contentType string, contentEncoding string, entityPointer interface{}) (err error) {
	var buffer []byte
	if r.bodyContent != nil {
		buffer = *r.bodyContent
	} else {
		buffer, err = ioutil.ReadAll(r.Request.Body)
		if err != nil {
			return err
		}
		r.bodyContent = &buffer
	}
	return r.decodeEntity(bytes.NewReader(buffer), contentType, contentEncoding, entityPointer)
}

func (r *Request) decodeEntity(reader io.Reader, contentType string, contentEncoding string, entityPointer interface{}) (err error) {
	entityReader := reader

	// check if the request body needs decompression
	if ENCODING_GZIP == contentEncoding {
		gzipReader := GzipReaderPool.Get().(*gzip.Reader)
		gzipReader.Reset(reader)
		entityReader = gzipReader
	} else if ENCODING_DEFLATE == contentEncoding {
		zlibReader, err := zlib.NewReader(reader)
		if err != nil {
			return err
		}
		entityReader = zlibReader
	}

	// decode JSON
	if strings.Contains(contentType, MIME_JSON) || MIME_JSON == defaultRequestContentType {
		decoder := json.NewDecoder(entityReader)
		decoder.UseNumber()
		return decoder.Decode(entityPointer)
	}

	// decode XML
	if strings.Contains(contentType, MIME_XML) || MIME_XML == defaultRequestContentType {
		return xml.NewDecoder(entityReader).Decode(entityPointer)
	}

	return NewError(http.StatusBadRequest, "Unable to unmarshal content of type:"+contentType)
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
