package calls

// go generate -import github.com/mesos/mesos-go/api/v1/lib/agent -type C:agent.Call
// GENERATED CODE FOLLOWS; DO NOT EDIT.

import (
	"context"

	"github.com/mesos/mesos-go/api/v1/lib"
	"github.com/mesos/mesos-go/api/v1/lib/encoding"

	"github.com/mesos/mesos-go/api/v1/lib/agent"
)

type (
	// Request generates a Call that's sent to a Mesos agent. Subsequent invocations are expected to
	// yield equivalent calls. Intended for use w/ non-streaming requests to an agent.
	Request interface {
		Call() *agent.Call
	}

	// RequestFunc is the functional adaptation of Request.
	RequestFunc func() *agent.Call

	// RequestStreaming generates a Call that's send to a Mesos agent. Subsequent invocations MAY generate
	// different Call objects. No more Call objects are expected once a nil is returned to signal the end of
	// of the request stream.
	RequestStreaming interface {
		Request
		IsStreaming()
	}

	// RequestStreamingFunc is the functional adaptation of RequestStreaming.
	RequestStreamingFunc func() *agent.Call

	// Send issues a Request to a Mesos agent and properly manages Call-specific mechanics.
	Sender interface {
		Send(context.Context, Request) (mesos.Response, error)
	}

	// SenderFunc is the functional adaptation of the Sender interface
	SenderFunc func(context.Context, Request) (mesos.Response, error)
)

func (f RequestFunc) Call() *agent.Call { return f() }

func (f RequestFunc) Marshaler() encoding.Marshaler {
	// avoid returning (*agent.Call)(nil) for interface type
	if call := f(); call != nil {
		return call
	}
	return nil
}

func (f RequestStreamingFunc) Push(c ...*agent.Call) RequestStreamingFunc { return Push(f, c...) }

func (f RequestStreamingFunc) Marshaler() encoding.Marshaler {
	// avoid returning (*agent.Call)(nil) for interface type
	if call := f(); call != nil {
		return call
	}
	return nil
}

func (f RequestStreamingFunc) IsStreaming() {}

func (f RequestStreamingFunc) Call() *agent.Call { return f() }

// Push prepends one or more calls onto a request stream. If no calls are given then the original stream is returned.
func Push(r RequestStreaming, c ...*agent.Call) RequestStreamingFunc {
	return func() *agent.Call {
		if len(c) == 0 {
			return r.Call()
		}
		head := c[0]
		c = c[1:]
		return head
	}
}

// Empty generates a stream that always returns nil.
func Empty() RequestStreamingFunc { return func() *agent.Call { return nil } }

var (
	_ = Request(RequestFunc(nil))
	_ = RequestStreaming(RequestStreamingFunc(nil))
	_ = Sender(SenderFunc(nil))
)

// NonStreaming returns a RequestFunc that always generates the same Call.
func NonStreaming(c *agent.Call) RequestFunc { return func() *agent.Call { return c } }

// FromChan returns a streaming request that fetches calls from the given channel until it closes.
// If a nil chan is specified then the returned func will always generate nil.
func FromChan(ch <-chan *agent.Call) RequestStreamingFunc {
	if ch == nil {
		// avoid blocking forever if we're handed a nil chan
		return func() *agent.Call { return nil }
	}
	return func() *agent.Call {
		if m, ok := <-ch; ok {
			return m
		}
		return nil
	}
}

// Send implements the Sender interface for SenderFunc
func (f SenderFunc) Send(ctx context.Context, r Request) (mesos.Response, error) {
	return f(ctx, r)
}

// IgnoreResponse generates a sender that closes any non-nil response received by Mesos.
func IgnoreResponse(s Sender) SenderFunc {
	return func(ctx context.Context, r Request) (mesos.Response, error) {
		resp, err := s.Send(ctx, r)
		if resp != nil {
			resp.Close()
		}
		return nil, err
	}
}

// SendNoData is a convenience func that executes the given Call using the provided Sender
// and always drops the response data.
func SendNoData(ctx context.Context, sender Sender, r Request) (err error) {
	_, err = IgnoreResponse(sender).Send(ctx, r)
	return
}
