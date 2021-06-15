package runtime

import (
	"context"
	"fmt"
	"net/http"
	"net/textproto"
	"strings"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// A HandlerFunc handles a specific pair of path pattern and HTTP method.
type HandlerFunc func(w http.ResponseWriter, r *http.Request, pathParams map[string]string)

// ErrUnknownURI is the error supplied to a custom ProtoErrorHandlerFunc when
// a request is received with a URI path that does not match any registered
// service method.
//
// Since gRPC servers return an "Unimplemented" code for requests with an
// unrecognized URI path, this error also has a gRPC "Unimplemented" code.
var ErrUnknownURI = status.Error(codes.Unimplemented, http.StatusText(http.StatusNotImplemented))

// ServeMux is a request multiplexer for grpc-gateway.
// It matches http requests to patterns and invokes the corresponding handler.
type ServeMux struct {
	// handlers maps HTTP method to a list of handlers.
	handlers                  map[string][]handler
	forwardResponseOptions    []func(context.Context, http.ResponseWriter, proto.Message) error
	marshalers                marshalerRegistry
	incomingHeaderMatcher     HeaderMatcherFunc
	outgoingHeaderMatcher     HeaderMatcherFunc
	metadataAnnotators        []func(context.Context, *http.Request) metadata.MD
	streamErrorHandler        StreamErrorHandlerFunc
	protoErrorHandler         ProtoErrorHandlerFunc
	disablePathLengthFallback bool
	lastMatchWins             bool
}

// ServeMuxOption is an option that can be given to a ServeMux on construction.
type ServeMuxOption func(*ServeMux)

// WithForwardResponseOption returns a ServeMuxOption representing the forwardResponseOption.
//
// forwardResponseOption is an option that will be called on the relevant context.Context,
// http.ResponseWriter, and proto.Message before every forwarded response.
//
// The message may be nil in the case where just a header is being sent.
func WithForwardResponseOption(forwardResponseOption func(context.Context, http.ResponseWriter, proto.Message) error) ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.forwardResponseOptions = append(serveMux.forwardResponseOptions, forwardResponseOption)
	}
}

// SetQueryParameterParser sets the query parameter parser, used to populate message from query parameters.
// Configuring this will mean the generated swagger output is no longer correct, and it should be
// done with careful consideration.
func SetQueryParameterParser(queryParameterParser QueryParameterParser) ServeMuxOption {
	return func(serveMux *ServeMux) {
		currentQueryParser = queryParameterParser
	}
}

// HeaderMatcherFunc checks whether a header key should be forwarded to/from gRPC context.
type HeaderMatcherFunc func(string) (string, bool)

// DefaultHeaderMatcher is used to pass http request headers to/from gRPC context. This adds permanent HTTP header
// keys (as specified by the IANA) to gRPC context with grpcgateway- prefix. HTTP headers that start with
// 'Grpc-Metadata-' are mapped to gRPC metadata after removing prefix 'Grpc-Metadata-'.
func DefaultHeaderMatcher(key string) (string, bool) {
	key = textproto.CanonicalMIMEHeaderKey(key)
	if isPermanentHTTPHeader(key) {
		return MetadataPrefix + key, true
	} else if strings.HasPrefix(key, MetadataHeaderPrefix) {
		return key[len(MetadataHeaderPrefix):], true
	}
	return "", false
}

// WithIncomingHeaderMatcher returns a ServeMuxOption representing a headerMatcher for incoming request to gateway.
//
// This matcher will be called with each header in http.Request. If matcher returns true, that header will be
// passed to gRPC context. To transform the header before passing to gRPC context, matcher should return modified header.
func WithIncomingHeaderMatcher(fn HeaderMatcherFunc) ServeMuxOption {
	return func(mux *ServeMux) {
		mux.incomingHeaderMatcher = fn
	}
}

// WithOutgoingHeaderMatcher returns a ServeMuxOption representing a headerMatcher for outgoing response from gateway.
//
// This matcher will be called with each header in response header metadata. If matcher returns true, that header will be
// passed to http response returned from gateway. To transform the header before passing to response,
// matcher should return modified header.
func WithOutgoingHeaderMatcher(fn HeaderMatcherFunc) ServeMuxOption {
	return func(mux *ServeMux) {
		mux.outgoingHeaderMatcher = fn
	}
}

// WithMetadata returns a ServeMuxOption for passing metadata to a gRPC context.
//
// This can be used by services that need to read from http.Request and modify gRPC context. A common use case
// is reading token from cookie and adding it in gRPC context.
func WithMetadata(annotator func(context.Context, *http.Request) metadata.MD) ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.metadataAnnotators = append(serveMux.metadataAnnotators, annotator)
	}
}

// WithProtoErrorHandler returns a ServeMuxOption for configuring a custom error handler.
//
// This can be used to handle an error as general proto message defined by gRPC.
// When this option is used, the mux uses the configured error handler instead of HTTPError and
// OtherErrorHandler.
func WithProtoErrorHandler(fn ProtoErrorHandlerFunc) ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.protoErrorHandler = fn
	}
}

// WithDisablePathLengthFallback returns a ServeMuxOption for disable path length fallback.
func WithDisablePathLengthFallback() ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.disablePathLengthFallback = true
	}
}

// WithStreamErrorHandler returns a ServeMuxOption that will use the given custom stream
// error handler, which allows for customizing the error trailer for server-streaming
// calls.
//
// For stream errors that occur before any response has been written, the mux's
// ProtoErrorHandler will be invoked. However, once data has been written, the errors must
// be handled differently: they must be included in the response body. The response body's
// final message will include the error details returned by the stream error handler.
func WithStreamErrorHandler(fn StreamErrorHandlerFunc) ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.streamErrorHandler = fn
	}
}

// WithLastMatchWins returns a ServeMuxOption that will enable "last
// match wins" behavior, where if multiple path patterns match a
// request path, the last one defined in the .proto file will be used.
func WithLastMatchWins() ServeMuxOption {
	return func(serveMux *ServeMux) {
		serveMux.lastMatchWins = true
	}
}

// NewServeMux returns a new ServeMux whose internal mapping is empty.
func NewServeMux(opts ...ServeMuxOption) *ServeMux {
	serveMux := &ServeMux{
		handlers:               make(map[string][]handler),
		forwardResponseOptions: make([]func(context.Context, http.ResponseWriter, proto.Message) error, 0),
		marshalers:             makeMarshalerMIMERegistry(),
		streamErrorHandler:     DefaultHTTPStreamErrorHandler,
	}

	for _, opt := range opts {
		opt(serveMux)
	}

	if serveMux.incomingHeaderMatcher == nil {
		serveMux.incomingHeaderMatcher = DefaultHeaderMatcher
	}

	if serveMux.outgoingHeaderMatcher == nil {
		serveMux.outgoingHeaderMatcher = func(key string) (string, bool) {
			return fmt.Sprintf("%s%s", MetadataHeaderPrefix, key), true
		}
	}

	return serveMux
}

// Handle associates "h" to the pair of HTTP method and path pattern.
func (s *ServeMux) Handle(meth string, pat Pattern, h HandlerFunc) {
	if s.lastMatchWins {
		s.handlers[meth] = append([]handler{handler{pat: pat, h: h}}, s.handlers[meth]...)
	} else {
		s.handlers[meth] = append(s.handlers[meth], handler{pat: pat, h: h})
	}
}

// ServeHTTP dispatches the request to the first handler whose pattern matches to r.Method and r.Path.
func (s *ServeMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	path := r.URL.Path
	if !strings.HasPrefix(path, "/") {
		if s.protoErrorHandler != nil {
			_, outboundMarshaler := MarshalerForRequest(s, r)
			sterr := status.Error(codes.InvalidArgument, http.StatusText(http.StatusBadRequest))
			s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, sterr)
		} else {
			OtherErrorHandler(w, r, http.StatusText(http.StatusBadRequest), http.StatusBadRequest)
		}
		return
	}

	components := strings.Split(path[1:], "/")
	l := len(components)
	var verb string
	if idx := strings.LastIndex(components[l-1], ":"); idx == 0 {
		if s.protoErrorHandler != nil {
			_, outboundMarshaler := MarshalerForRequest(s, r)
			s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, ErrUnknownURI)
		} else {
			OtherErrorHandler(w, r, http.StatusText(http.StatusNotFound), http.StatusNotFound)
		}
		return
	} else if idx > 0 {
		c := components[l-1]
		components[l-1], verb = c[:idx], c[idx+1:]
	}

	if override := r.Header.Get("X-HTTP-Method-Override"); override != "" && s.isPathLengthFallback(r) {
		r.Method = strings.ToUpper(override)
		if err := r.ParseForm(); err != nil {
			if s.protoErrorHandler != nil {
				_, outboundMarshaler := MarshalerForRequest(s, r)
				sterr := status.Error(codes.InvalidArgument, err.Error())
				s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, sterr)
			} else {
				OtherErrorHandler(w, r, err.Error(), http.StatusBadRequest)
			}
			return
		}
	}
	for _, h := range s.handlers[r.Method] {
		pathParams, err := h.pat.Match(components, verb)
		if err != nil {
			continue
		}
		h.h(w, r, pathParams)
		return
	}

	// lookup other methods to handle fallback from GET to POST and
	// to determine if it is MethodNotAllowed or NotFound.
	for m, handlers := range s.handlers {
		if m == r.Method {
			continue
		}
		for _, h := range handlers {
			pathParams, err := h.pat.Match(components, verb)
			if err != nil {
				continue
			}
			// X-HTTP-Method-Override is optional. Always allow fallback to POST.
			if s.isPathLengthFallback(r) {
				if err := r.ParseForm(); err != nil {
					if s.protoErrorHandler != nil {
						_, outboundMarshaler := MarshalerForRequest(s, r)
						sterr := status.Error(codes.InvalidArgument, err.Error())
						s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, sterr)
					} else {
						OtherErrorHandler(w, r, err.Error(), http.StatusBadRequest)
					}
					return
				}
				h.h(w, r, pathParams)
				return
			}
			if s.protoErrorHandler != nil {
				_, outboundMarshaler := MarshalerForRequest(s, r)
				s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, ErrUnknownURI)
			} else {
				OtherErrorHandler(w, r, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
			}
			return
		}
	}

	if s.protoErrorHandler != nil {
		_, outboundMarshaler := MarshalerForRequest(s, r)
		s.protoErrorHandler(ctx, s, outboundMarshaler, w, r, ErrUnknownURI)
	} else {
		OtherErrorHandler(w, r, http.StatusText(http.StatusNotFound), http.StatusNotFound)
	}
}

// GetForwardResponseOptions returns the ForwardResponseOptions associated with this ServeMux.
func (s *ServeMux) GetForwardResponseOptions() []func(context.Context, http.ResponseWriter, proto.Message) error {
	return s.forwardResponseOptions
}

func (s *ServeMux) isPathLengthFallback(r *http.Request) bool {
	return !s.disablePathLengthFallback && r.Method == "POST" && r.Header.Get("Content-Type") == "application/x-www-form-urlencoded"
}

type handler struct {
	pat Pattern
	h   HandlerFunc
}
