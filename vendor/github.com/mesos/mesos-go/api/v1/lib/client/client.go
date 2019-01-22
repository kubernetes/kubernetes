package client

import (
	"github.com/mesos/mesos-go/api/v1/lib/encoding"
)

type (
	// ResponseClass indicates the kind of response that a caller is expecting from Mesos.
	ResponseClass int

	// Request is a non-streaming request from the client to the server.
	// Marshaler always returns the same object; the object is sent once to the server and then
	// a response is expected.
	Request interface {
		Marshaler() encoding.Marshaler
	}

	// RequestStreaming is a streaming request from the client to the server.
	// Marshaler returns a new object for upon each invocation, nil when there are no more objects to send.
	// Client implementations are expected to differentiate between Request and RequestStreaming either by
	// type-switching or by attempting interface conversion.
	RequestStreaming interface {
		Request
		IsStreaming()
	}

	RequestFunc          func() encoding.Marshaler
	RequestStreamingFunc func() encoding.Marshaler
)

var (
	_ = Request(RequestFunc(nil))
	_ = RequestStreaming(RequestStreamingFunc(nil))
)

func (f RequestFunc) Marshaler() encoding.Marshaler          { return f() }
func (f RequestStreamingFunc) Marshaler() encoding.Marshaler { return f() }
func (f RequestStreamingFunc) IsStreaming()                  {}

// RequestSingleton generates a non-streaming Request that always returns the same marshaler
func RequestSingleton(m encoding.Marshaler) Request {
	return RequestFunc(func() encoding.Marshaler { return m })
}

const (
	ResponseClassSingleton ResponseClass = iota
	ResponseClassStreaming
	ResponseClassNoData

	// ResponseClassAuto should be used with versions of Mesos prior to 1.2.x.
	// Otherwise, this type is deprecated and callers should use ResponseClassSingleton
	// or ResponseClassStreaming instead.
	ResponseClassAuto
)
