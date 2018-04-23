/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package filters

import (
	"compress/gzip"
	"compress/zlib"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/emicklei/go-restful"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// Compressor is an interface to compression writers
type Compressor interface {
	io.WriteCloser
	Flush() error
}

const (
	headerAcceptEncoding  = "Accept-Encoding"
	headerContentEncoding = "Content-Encoding"

	encodingGzip    = "gzip"
	encodingDeflate = "deflate"
)

// WithCompression wraps an http.Handler with the Compression Handler
func WithCompression(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		wantsCompression, encoding := wantsCompressedResponse(req)
		w.Header().Set("Vary", "Accept-Encoding")
		if wantsCompression {
			compressionWriter, err := NewCompressionResponseWriter(w, encoding)
			if err != nil {
				handleError(w, req, err)
				runtime.HandleError(fmt.Errorf("failed to compress HTTP response: %v", err))
				return
			}
			compressionWriter.Header().Set("Content-Encoding", encoding)
			handler.ServeHTTP(compressionWriter, req)
			compressionWriter.(*compressionResponseWriter).Close()
		} else {
			handler.ServeHTTP(w, req)
		}
	})
}

// wantsCompressedResponse reads the Accept-Encoding header to see if and which encoding is requested.
func wantsCompressedResponse(req *http.Request) (bool, string) {
	// don't compress watches
	ctx := req.Context()
	info, ok := request.RequestInfoFrom(ctx)
	if !ok {
		return false, ""
	}
	if !info.IsResourceRequest {
		return false, ""
	}
	if info.Verb == "watch" {
		return false, ""
	}
	header := req.Header.Get(headerAcceptEncoding)
	gi := strings.Index(header, encodingGzip)
	zi := strings.Index(header, encodingDeflate)
	// use in order of appearance
	switch {
	case gi == -1:
		return zi != -1, encodingDeflate
	case zi == -1:
		return gi != -1, encodingGzip
	case gi < zi:
		return true, encodingGzip
	default:
		return true, encodingDeflate
	}
}

type compressionResponseWriter struct {
	writer     http.ResponseWriter
	compressor Compressor
	encoding   string
}

// NewCompressionResponseWriter returns wraps w with a compression ResponseWriter, using the given encoding
func NewCompressionResponseWriter(w http.ResponseWriter, encoding string) (http.ResponseWriter, error) {
	var compressor Compressor
	switch encoding {
	case encodingGzip:
		compressor = gzip.NewWriter(w)
	case encodingDeflate:
		compressor = zlib.NewWriter(w)
	default:
		return nil, fmt.Errorf("%s is not a supported encoding type", encoding)
	}
	return &compressionResponseWriter{
		writer:     w,
		compressor: compressor,
		encoding:   encoding,
	}, nil
}

// compressionResponseWriter implements http.Responsewriter Interface
var _ http.ResponseWriter = &compressionResponseWriter{}

func (c *compressionResponseWriter) Header() http.Header {
	return c.writer.Header()
}

// compress data according to compression method
func (c *compressionResponseWriter) Write(p []byte) (int, error) {
	if c.compressorClosed() {
		return -1, errors.New("compressing error: tried to write data using closed compressor")
	}
	c.Header().Set(headerContentEncoding, c.encoding)
	defer c.compressor.Flush()
	return c.compressor.Write(p)
}

func (c *compressionResponseWriter) WriteHeader(status int) {
	c.writer.WriteHeader(status)
}

// CloseNotify is part of http.CloseNotifier interface
func (c *compressionResponseWriter) CloseNotify() <-chan bool {
	return c.writer.(http.CloseNotifier).CloseNotify()
}

// Close the underlying compressor
func (c *compressionResponseWriter) Close() error {
	if c.compressorClosed() {
		return errors.New("Compressing error: tried to close already closed compressor")
	}

	c.compressor.Close()
	c.compressor = nil
	return nil
}

func (c *compressionResponseWriter) Flush() {
	if c.compressorClosed() {
		return
	}
	c.compressor.Flush()
}

func (c *compressionResponseWriter) compressorClosed() bool {
	return nil == c.compressor
}

// RestfulWithCompression wraps WithCompression to be compatible with go-restful
func RestfulWithCompression(function restful.RouteFunction) restful.RouteFunction {
	return restful.RouteFunction(func(request *restful.Request, response *restful.Response) {
		handler := WithCompression(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			response.ResponseWriter = w
			request.Request = req
			function(request, response)
		}))
		handler.ServeHTTP(response.ResponseWriter, request.Request)
	})
}
