/*
Copyright 2024 The Kubernetes Authors.

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

package restproxy

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/go-logr/logr"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	restproxyapi "k8s.io/dynamic-resource-allocation/apis/restproxy/v1alpha1"
	"k8s.io/klog/v2"
)

// StartRoundTripper creates a new client. [RoundTrip] can be used right away, but
// will not be able to transmit data unless the server contacts the client
// through [NodeStartRESTProxy] and establishes a stream.
//
// Canceling the context stops all background activity. [WaitForShutdown] can
// be used block until all goroutines have stopped after cancellation.
func StartRoundTripper(ctx context.Context) *RoundTripper {
	logger := klog.FromContext(ctx)
	ctx, cancel := context.WithCancelCause(ctx)
	rt := &RoundTripper{
		ctx:     ctx,
		cancel:  cancel,
		streams: make(map[int64]*stream),
	}
	rt.cond = sync.NewCond(&rt.mutex)

	logger.V(3).Info("Starting")
	rt.wg.Add(1)
	go func() {
		defer rt.wg.Done()
		defer logger.V(3).Info("Stopping")
		defer rt.cancel(errors.New("RoundTripper stopped"))
		defer utilruntime.HandleCrash()
		<-ctx.Done()

		// Notify goroutines which are blocked on c.cond.Wait.
		// All goroutines must check for cancellation before
		// calling Wait.
		rt.mutex.Lock()
		defer rt.mutex.Unlock()
		rt.cond.Broadcast()
	}()
	return rt
}

var _ http.RoundTripper = &RoundTripper{}
var _ restproxyapi.RESTServer = &RoundTripper{}

// RoundTripper is sending REST requests to the server via the proxy.
// It implements [http.RoundTripper].
type RoundTripper struct {
	ctx    context.Context
	cancel func(cause error)
	mutex  sync.Mutex
	cond   *sync.Cond
	wg     sync.WaitGroup

	// idCounter continuously gets incremented for each new request.
	idCounter int64

	// proxyServer is nil if we don't have an active stream to a proxy.
	proxyServer restproxyapi.REST_ProxyServer

	// streams contains all active streams, indexed by their ID.
	streams map[int64]*stream
}

type stream struct {
	logger klog.Logger
	id     int64

	// responseHeader is nil while we have no response yet.
	responseHeader *restproxyapi.ResponseHeader

	// responseData is the next chunk of the response body,
	// if there is one.
	responseData *chunk

	// read is the total amount of data consumed from the response data.
	// There's no guarantee that chunks get stored in order, so [bodyReader.Read]
	// has to check that the next data is what it needs.
	read int64

	// errMsg is the error string that shall be returned by [bodyReader.Read]
	// after delivering the response data.
	errMsg string

	// closed indicates that no more data is coming after what was sent so
	// far or that the reader is no longer interested in receiving more data.
	closed bool
}

// MarshalLog logs the struct without any pointers and without data.
func (s *stream) MarshalLog() any {
	obj := map[string]any{
		"id":             s.id,
		"responseHeader": s.responseHeader,
		"responseData": func() any {
			if s.responseData == nil {
				return nil
			}
			return map[string]int64{
				"offset": s.responseData.offset,
				"size":   int64(len(s.responseData.data)),
			}
		}(),
		"read":   s.read,
		"errMsg": s.errMsg,
		"closed": s.closed,
	}
	return obj
}

var _ logr.Marshaler = &stream{}

type chunk struct {
	offset int64
	data   []byte
}

// Stop causes all goroutines to stop immediately and waits for them to stop.
// Canceling the context passed to Start also causes goroutines to stop.
func (rt *RoundTripper) Stop() {
	rt.cancel(errors.New("RoundTripper stopping"))
	rt.wg.Wait()
}

// NewRESTConfig returns a new config which uses the REST proxy.
func (rt *RoundTripper) NewRESTConfig() *rest.Config {
	config := &rest.Config{}
	config.Wrap(rt.wrap)
	return config
}

func (rt *RoundTripper) wrap(_ http.RoundTripper) http.RoundTripper {
	// Instead of wrapping, replace it.
	return rt
}

// Proxy implements [RESTServer.Proxy].
//
// It stores the server stream and thus enables sending REST requests to the
// proxy through that stream.
func (rt *RoundTripper) Proxy(_ *restproxyapi.ProxyMessage, server restproxyapi.REST_ProxyServer) error {
	logger := klog.FromContext(rt.ctx)

	func() {
		rt.mutex.Lock()
		defer rt.mutex.Unlock()

		if rt.proxyServer != nil {
			// Called again? Close all previous requests.
			logger.V(3).Info("Closing all streams, Proxy got called again")
			rt.streams = make(map[int64]*stream)
		}

		rt.proxyServer = server
		rt.cond.Broadcast()
		logger.V(2).Info("Started client of REST proxy")
	}()

	// We need to block until it is time to shut down because returning
	// would close the stream from our end.
	<-rt.ctx.Done()
	logger.V(3).Info("Stopped client of REST proxy")

	return nil
}

// Reply implement [RESTServer.Reply].
//
// It receives more information about the response to a previous request and
// updates the response stream with it. It blocks if response body data is
// delivered faster than it gets consumed.
func (rt *RoundTripper) Reply(ctx context.Context, reply *restproxyapi.ReplyMessage) (*restproxyapi.ReplyResponse, error) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()
	stream := rt.streams[reply.Id]

	logger := klog.FromContext(ctx)
	if stream == nil {
		logger.V(5).Info("Closing REST proxy response stream, it is gone", "requestID", reply.Id)
		return &restproxyapi.ReplyResponse{
			Close: true,
		}, nil
	}

	if stream.responseHeader == nil && reply.Header != nil {
		header := *reply.Header
		stream.responseHeader = &header
	}

	// If there is already some data, then block until it is read before
	// making the next chunk available. This then also blocks reading
	// more data in the REST proxy because it waits for our reply.
	//
	// This has to be done before setting the error or close flag,
	// otherwise the reader might see those before we get a chance to add
	// the final data chunk here.
	if len(reply.Body) > 0 {
		for stream.responseData != nil && ctx.Err() == nil {
			rt.cond.Wait()
		}
		if ctx.Err() != nil {
			// We cannot count on the REST proxy trying again,
			// so mark the stream as failed.
			stream.errMsg = fmt.Sprintf("delivering response body data: %v", context.Cause(ctx))
			rt.cond.Broadcast()
			return nil, errors.New(stream.errMsg)
		}
		stream.responseData = &chunk{offset: reply.BodyOffset, data: reply.Body}
	}

	if reply.Error != "" {
		stream.errMsg = reply.Error
	}
	if reply.Close {
		stream.closed = true
	}

	rt.cond.Broadcast()

	return &restproxyapi.ReplyResponse{
		// This may also be set at our end.
		Close: stream.closed,
	}, nil
}

// NodeObject implements [RESTServer.NodeObject].
//
// This information is not relevant for the roundtripper itself.
// Whoever needs it can wrap the RoundTripper.
func (rt *RoundTripper) NodeObject(ctx context.Context, req *restproxyapi.NodeObjectRequest) (*restproxyapi.NodeObjectResponse, error) {
	return &restproxyapi.NodeObjectResponse{}, nil
}

// RoundTrip implements [http.RoundTripper.Roundtrip].
//
// Only the normal situation that the REST proxy has not connected yet
// is handled transparently. Unexpected errors get returned, with
// retries in the caller.
func (rt *RoundTripper) RoundTrip(req *http.Request) (finalResp *http.Response, finalErr error) {
	logger := klog.FromContext(rt.ctx)
	id := int64(0)

	if req.URL == nil {
		return nil, errors.New("URL is required")
	}

	// Retrieve the full body before locking the mutex.
	var body []byte
	if req.Body != nil {
		b, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("reading request body: %w", err)
		}
		if err := req.Body.Close(); err != nil {
			return nil, fmt.Errorf("closing request body: %w", err)
		}
		body = b
	}

	rt.mutex.Lock()
	defer func() {
		defer rt.mutex.Unlock()

		// Always remove the stream when we are not going to read from it.
		if finalErr != nil {
			logger.V(5).Info("Closing stream", "err", finalErr)
			delete(rt.streams, id)
			rt.cond.Broadcast()
		} else {
			logger.V(5).Info("Response body is ready for reading")
		}
	}()

	// Generate new stream.
	rt.idCounter++
	id = rt.idCounter
	stream := &stream{
		logger: logger,
		id:     id,
	}
	rt.streams[id] = stream

	proxyHeader := make(map[string]*restproxyapi.RESTHeader, len(req.Header))
	for name, values := range req.Header {
		proxyHeader[name] = &restproxyapi.RESTHeader{Values: values}
	}
	proxyReq := &restproxyapi.Request{
		Id:       stream.id,
		Method:   req.Method,
		Path:     req.URL.Path,
		Header:   proxyHeader,
		RawQuery: req.URL.RawQuery,
		Body:     body,
	}

	// Log full details once, then only the ID.
	logger.V(5).Info("New REST request", "request", logRequest{logger: &logger, request: proxyReq})
	logger = klog.LoggerWithValues(logger, "request", logRequest{request: proxyReq})

	// Determine whether the request has a context or cancel stream.
	// A goroutine will block on those and notify the parent by setting
	// "canceled" when cancellation happened, while holding rt.mutex.
	var canceled error
	reqCtx := req.Context()
	// SA1019: req.Cancel has been deprecated since Go 1.7: Set the Request's context with NewRequestWithContext instead. If a Request's Cancel field and context are both set, it is undefined whether Cancel is respected.
	// The comment about deprecation is correct. But here we don't *set* req.Cancel,
	// we check for it because it is unknown whether the caller might still be using
	// it.
	//nolint:staticcheck
	if reqCtx == nil && req.Cancel != nil {
		reqCtx = wait.ContextForChannel(req.Cancel)
	}
	if reqCtx != nil {
		// Always stop the goroutine when the request is dealt with.
		reqCtx, cancel := context.WithCancel(reqCtx)
		defer cancel()

		rt.wg.Add(1)
		go func() {
			defer rt.wg.Done()
			defer utilruntime.HandleCrash()

			var why error
			select {
			case <-reqCtx.Done():
				why = context.Cause(reqCtx)
			case <-rt.ctx.Done():
				why = context.Cause(rt.ctx)
			}

			rt.mutex.Lock()
			defer rt.mutex.Unlock()
			canceled = why
			rt.cond.Broadcast()
		}()
	}

	// Wait for proxy.
	for rt.proxyServer == nil {
		if err := context.Cause(rt.ctx); err != nil {
			return nil, fmt.Errorf("waiting for proxy server failed: %w", err)
		}
		rt.cond.Wait()
	}

	// Tell the proxy to start the request.
	//
	// It is not safe to call Send on the same stream in different
	// goroutines, so we have to hold this lock while sending.
	if err := rt.proxyServer.Send(proxyReq); err != nil {
		return nil, fmt.Errorf("sending proxy request: %w", err)
	}

	// Response information and data will be received by NodeRESTReply
	// and gets stored in the stream. From there the body reader takes
	// as much data as it can until the stream is closed or failed.
	// But first we need the initial response.
	for {
		stream := rt.streams[id]
		if stream == nil {
			// Stream got removed.
			return nil, fmt.Errorf("stream %d got removed", id)
		}

		if stream.responseHeader != nil {
			header := make(http.Header, len(stream.responseHeader.Header))
			for name, values := range stream.responseHeader.Header {
				header[name] = values.Values
			}
			req.Body = nil
			resp := &http.Response{
				Status:     stream.responseHeader.Status,
				StatusCode: int(stream.responseHeader.StatusCode),
				Proto:      stream.responseHeader.Proto,
				ProtoMajor: int(stream.responseHeader.ProtoMajor),
				ProtoMinor: int(stream.responseHeader.ProtoMinor),
				Header:     header,
				Body: bodyReader{
					reqCtx:       reqCtx,
					logger:       logger,
					roundTripper: rt,
					id:           id,
				},
				Request: req,
			}
			return resp, nil
		}

		if stream.errMsg != "" {
			// Some error occurred and we didn't even get the header.
			return nil, fmt.Errorf("REST proxy reported an error: %s", stream.errMsg)
		}

		// canceled only gets set when the request has a context or channel.
		if canceled != nil {
			return nil, fmt.Errorf("REST request canceled: %w", canceled)
		}

		// This context always exists.
		if err := context.Cause(rt.ctx); err != nil {
			return nil, fmt.Errorf("REST proxy canceled: %w", err)
		}
		rt.cond.Wait()
	}
}

// bodyReader holds a reference to the client and reads response data for one
// stream.
type bodyReader struct {
	reqCtx       context.Context
	logger       klog.Logger
	roundTripper *RoundTripper
	id           int64
}

func (b bodyReader) Read(p []byte) (read int, finalErr error) {
	rt := b.roundTripper
	logger := b.logger
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	logger.V(5).Info("Reading body", "size", len(p))
	defer func() {
		if finalErr != nil {
			stream := rt.streams[b.id]
			if stream != nil {
				stream.closed = true
			}
		}
		// We must have changed something, so wake up the gRPC worker.
		rt.cond.Broadcast()
		logger.V(5).Info("Read body", "len", read, "err", finalErr)
	}()

	for {
		stream := rt.streams[b.id]
		if stream == nil {
			// Stream got removed.
			return 0, fmt.Errorf("stream %d got removed", b.id)
		}

		if stream.responseData != nil {
			start := int(stream.read - stream.responseData.offset)
			remaining := len(stream.responseData.data) - start
			n := len(p)
			if n > remaining {
				n = remaining
			}
			copy(p, stream.responseData.data[start:start+n])
			if start+n == len(stream.responseData.data) {
				// Done with the next chunk.
				stream.responseData = nil
			}
			stream.read += int64(n)
			logger.V(5).Info("Updated stream after reading from response", "n", n, "stream", stream)
			return n, nil
		}

		if stream.errMsg != "" {
			return 0, errors.New(stream.errMsg)
		}

		if stream.closed {
			// All data consumed without an error.
			return 0, io.EOF
		}

		// TODO: should the request context really affect reading the body?
		// The Golang documentation doesn't say. Let's mimick the behavior
		// of the default transport.
		if b.reqCtx != nil {
			if err := context.Cause(b.reqCtx); err != nil {
				return 0, fmt.Errorf("REST request canceled: %w", err)
			}
		}

		if err := context.Cause(rt.ctx); err != nil {
			return 0, fmt.Errorf("REST proxy canceled: %w", err)
		}

		rt.cond.Wait()
	}
}

// Close removes the stream, which indicates to the proxy that no further data is
// needed if it tries to add more.
func (b bodyReader) Close() error {
	rt := b.roundTripper
	logger := b.logger
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	delete(rt.streams, b.id)
	logger.V(5).Info("Closing REST stream because response body got closed")
	rt.cond.Broadcast()
	return nil
}
