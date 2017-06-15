package opentracing

import (
	"errors"
	"net/http"
)

///////////////////////////////////////////////////////////////////////////////
// CORE PROPAGATION INTERFACES:
///////////////////////////////////////////////////////////////////////////////

var (
	// ErrUnsupportedFormat occurs when the `format` passed to Tracer.Inject() or
	// Tracer.Extract() is not recognized by the Tracer implementation.
	ErrUnsupportedFormat = errors.New("opentracing: Unknown or unsupported Inject/Extract format")

	// ErrSpanContextNotFound occurs when the `carrier` passed to
	// Tracer.Extract() is valid and uncorrupted but has insufficient
	// information to extract a SpanContext.
	ErrSpanContextNotFound = errors.New("opentracing: SpanContext not found in Extract carrier")

	// ErrInvalidSpanContext errors occur when Tracer.Inject() is asked to
	// operate on a SpanContext which it is not prepared to handle (for
	// example, since it was created by a different tracer implementation).
	ErrInvalidSpanContext = errors.New("opentracing: SpanContext type incompatible with tracer")

	// ErrInvalidCarrier errors occur when Tracer.Inject() or Tracer.Extract()
	// implementations expect a different type of `carrier` than they are
	// given.
	ErrInvalidCarrier = errors.New("opentracing: Invalid Inject/Extract carrier")

	// ErrSpanContextCorrupted occurs when the `carrier` passed to
	// Tracer.Extract() is of the expected type but is corrupted.
	ErrSpanContextCorrupted = errors.New("opentracing: SpanContext data corrupted in Extract carrier")
)

///////////////////////////////////////////////////////////////////////////////
// BUILTIN PROPAGATION FORMATS:
///////////////////////////////////////////////////////////////////////////////

// BuiltinFormat is used to demarcate the values within package `opentracing`
// that are intended for use with the Tracer.Inject() and Tracer.Extract()
// methods.
type BuiltinFormat byte

const (
	// Binary represents SpanContexts as opaque binary data.
	//
	// For Tracer.Inject(): the carrier must be an `io.Writer`.
	//
	// For Tracer.Extract(): the carrier must be an `io.Reader`.
	Binary BuiltinFormat = iota

	// TextMap represents SpanContexts as key:value string pairs.
	//
	// Unlike HTTPHeaders, the TextMap format does not restrict the key or
	// value character sets in any way.
	//
	// For Tracer.Inject(): the carrier must be a `TextMapWriter`.
	//
	// For Tracer.Extract(): the carrier must be a `TextMapReader`.
	TextMap

	// HTTPHeaders represents SpanContexts as HTTP header string pairs.
	//
	// Unlike TextMap, the HTTPHeaders format requires that the keys and values
	// be valid as HTTP headers as-is (i.e., character casing may be unstable
	// and special characters are disallowed in keys, values should be
	// URL-escaped, etc).
	//
	// For Tracer.Inject(): the carrier must be a `TextMapWriter`.
	//
	// For Tracer.Extract(): the carrier must be a `TextMapReader`.
	//
	// See HTTPHeadersCarrier for an implementation of both TextMapWriter
	// and TextMapReader that defers to an http.Header instance for storage.
	// For example, Inject():
	//
	//    carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
	//    err := span.Tracer().Inject(
	//        span.Context(), opentracing.HTTPHeaders, carrier)
	//
	// Or Extract():
	//
	//    carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
	//    clientContext, err := tracer.Extract(
	//        opentracing.HTTPHeaders, carrier)
	//
	HTTPHeaders
)

// TextMapWriter is the Inject() carrier for the TextMap builtin format. With
// it, the caller can encode a SpanContext for propagation as entries in a map
// of unicode strings.
type TextMapWriter interface {
	// Set a key:value pair to the carrier. Multiple calls to Set() for the
	// same key leads to undefined behavior.
	//
	// NOTE: The backing store for the TextMapWriter may contain data unrelated
	// to SpanContext. As such, Inject() and Extract() implementations that
	// call the TextMapWriter and TextMapReader interfaces must agree on a
	// prefix or other convention to distinguish their own key:value pairs.
	Set(key, val string)
}

// TextMapReader is the Extract() carrier for the TextMap builtin format. With it,
// the caller can decode a propagated SpanContext as entries in a map of
// unicode strings.
type TextMapReader interface {
	// ForeachKey returns TextMap contents via repeated calls to the `handler`
	// function. If any call to `handler` returns a non-nil error, ForeachKey
	// terminates and returns that error.
	//
	// NOTE: The backing store for the TextMapReader may contain data unrelated
	// to SpanContext. As such, Inject() and Extract() implementations that
	// call the TextMapWriter and TextMapReader interfaces must agree on a
	// prefix or other convention to distinguish their own key:value pairs.
	//
	// The "foreach" callback pattern reduces unnecessary copying in some cases
	// and also allows implementations to hold locks while the map is read.
	ForeachKey(handler func(key, val string) error) error
}

// TextMapCarrier allows the use of regular map[string]string
// as both TextMapWriter and TextMapReader.
type TextMapCarrier map[string]string

// ForeachKey conforms to the TextMapReader interface.
func (c TextMapCarrier) ForeachKey(handler func(key, val string) error) error {
	for k, v := range c {
		if err := handler(k, v); err != nil {
			return err
		}
	}
	return nil
}

// Set implements Set() of opentracing.TextMapWriter
func (c TextMapCarrier) Set(key, val string) {
	c[key] = val
}

// HTTPHeadersCarrier satisfies both TextMapWriter and TextMapReader.
//
// Example usage for server side:
//
//     carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
//     clientContext, err := tracer.Extract(opentracing.HTTPHeaders, carrier)
//
// Example usage for client side:
//
//     carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
//     err := tracer.Inject(
//         span.Context(),
//         opentracing.HTTPHeaders,
//         carrier)
//
type HTTPHeadersCarrier http.Header

// Set conforms to the TextMapWriter interface.
func (c HTTPHeadersCarrier) Set(key, val string) {
	h := http.Header(c)
	h.Add(key, val)
}

// ForeachKey conforms to the TextMapReader interface.
func (c HTTPHeadersCarrier) ForeachKey(handler func(key, val string) error) error {
	for k, vals := range c {
		for _, v := range vals {
			if err := handler(k, v); err != nil {
				return err
			}
		}
	}
	return nil
}
