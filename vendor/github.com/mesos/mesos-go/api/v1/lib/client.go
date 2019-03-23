package mesos

// DEPRECATED in favor of github.com/mesos/mesos-go/api/v1/lib/client

import (
	"io"

	"github.com/mesos/mesos-go/api/v1/lib/encoding"
)

// A Client represents a Mesos API client which can send Calls and return
// a streaming Decoder from which callers can read Events from, an io.Closer to
// close the event stream on graceful termination and an error in case of failure.
type Client interface {
	Do(encoding.Marshaler) (Response, error)
}

// ClientFunc is a functional adapter of the Client interface
type ClientFunc func(encoding.Marshaler) (Response, error)

// Do implements Client
func (cf ClientFunc) Do(m encoding.Marshaler) (Response, error) { return cf(m) }

// Response captures the output of a Mesos API operation. Callers are responsible for invoking
// Close when they're finished processing the response otherwise there may be connection leaks.
type Response interface {
	io.Closer
	encoding.Decoder
}

// ResponseDecorator optionally modifies the behavior of a Response
type ResponseDecorator interface {
	Decorate(Response) Response
}

// ResponseDecoratorFunc is the functional adapter for ResponseDecorator
type ResponseDecoratorFunc func(Response) Response

func (f ResponseDecoratorFunc) Decorate(r Response) Response { return f(r) }

// CloseFunc is the functional adapter for io.Closer
type CloseFunc func() error

// Close implements io.Closer
func (f CloseFunc) Close() error { return f() }

// ResponseWrapper delegates to optional overrides for invocations of Response methods.
type ResponseWrapper struct {
	Response Response
	Closer   io.Closer
	Decoder  encoding.Decoder
}

func (wrapper *ResponseWrapper) Close() error {
	if wrapper.Closer != nil {
		return wrapper.Closer.Close()
	}
	if wrapper.Response != nil {
		return wrapper.Response.Close()
	}
	return nil
}

func (wrapper *ResponseWrapper) Decode(u encoding.Unmarshaler) error {
	if wrapper.Decoder != nil {
		return wrapper.Decoder.Decode(u)
	}
	return wrapper.Response.Decode(u)
}

var _ = Response(&ResponseWrapper{})
