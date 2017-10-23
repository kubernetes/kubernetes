/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"bytes"
	"errors"
	"io"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

// StreamHandler defines the handler called by gRPC server to complete the
// execution of a streaming RPC.
type StreamHandler func(srv interface{}, stream ServerStream) error

// StreamDesc represents a streaming RPC service's method specification.
type StreamDesc struct {
	StreamName string
	Handler    StreamHandler

	// At least one of these is true.
	ServerStreams bool
	ClientStreams bool
}

// Stream defines the common interface a client or server stream has to satisfy.
type Stream interface {
	// Context returns the context for this stream.
	Context() context.Context
	// SendMsg blocks until it sends m, the stream is done or the stream
	// breaks.
	// On error, it aborts the stream and returns an RPC status on client
	// side. On server side, it simply returns the error to the caller.
	// SendMsg is called by generated code. Also Users can call SendMsg
	// directly when it is really needed in their use cases.
	SendMsg(m interface{}) error
	// RecvMsg blocks until it receives a message or the stream is
	// done. On client side, it returns io.EOF when the stream is done. On
	// any other error, it aborts the stream and returns an RPC status. On
	// server side, it simply returns the error to the caller.
	RecvMsg(m interface{}) error
}

// ClientStream defines the interface a client stream has to satisfy.
type ClientStream interface {
	// Header returns the header metadata received from the server if there
	// is any. It blocks if the metadata is not ready to read.
	Header() (metadata.MD, error)
	// Trailer returns the trailer metadata from the server, if there is any.
	// It must only be called after stream.CloseAndRecv has returned, or
	// stream.Recv has returned a non-nil error (including io.EOF).
	Trailer() metadata.MD
	// CloseSend closes the send direction of the stream. It closes the stream
	// when non-nil error is met.
	CloseSend() error
	Stream
}

// NewClientStream creates a new Stream for the client side. This is called
// by generated code.
func NewClientStream(ctx context.Context, desc *StreamDesc, cc *ClientConn, method string, opts ...CallOption) (_ ClientStream, err error) {
	if cc.dopts.streamInt != nil {
		return cc.dopts.streamInt(ctx, desc, cc, method, newClientStream, opts...)
	}
	return newClientStream(ctx, desc, cc, method, opts...)
}

func newClientStream(ctx context.Context, desc *StreamDesc, cc *ClientConn, method string, opts ...CallOption) (_ ClientStream, err error) {
	var (
		t      transport.ClientTransport
		s      *transport.Stream
		put    func()
		cancel context.CancelFunc
	)
	c := defaultCallInfo
	if mc, ok := cc.getMethodConfig(method); ok {
		c.failFast = !mc.WaitForReady
		if mc.Timeout > 0 {
			ctx, cancel = context.WithTimeout(ctx, mc.Timeout)
		}
	}
	for _, o := range opts {
		if err := o.before(&c); err != nil {
			return nil, toRPCErr(err)
		}
	}
	callHdr := &transport.CallHdr{
		Host:   cc.authority,
		Method: method,
		Flush:  desc.ServerStreams && desc.ClientStreams,
	}
	if cc.dopts.cp != nil {
		callHdr.SendCompress = cc.dopts.cp.Type()
	}
	var trInfo traceInfo
	if EnableTracing {
		trInfo.tr = trace.New("grpc.Sent."+methodFamily(method), method)
		trInfo.firstLine.client = true
		if deadline, ok := ctx.Deadline(); ok {
			trInfo.firstLine.deadline = deadline.Sub(time.Now())
		}
		trInfo.tr.LazyLog(&trInfo.firstLine, false)
		ctx = trace.NewContext(ctx, trInfo.tr)
		defer func() {
			if err != nil {
				// Need to call tr.finish() if error is returned.
				// Because tr will not be returned to caller.
				trInfo.tr.LazyPrintf("RPC: [%v]", err)
				trInfo.tr.SetError()
				trInfo.tr.Finish()
			}
		}()
	}
	ctx = newContextWithRPCInfo(ctx)
	sh := cc.dopts.copts.StatsHandler
	if sh != nil {
		ctx = sh.TagRPC(ctx, &stats.RPCTagInfo{FullMethodName: method})
		begin := &stats.Begin{
			Client:    true,
			BeginTime: time.Now(),
			FailFast:  c.failFast,
		}
		sh.HandleRPC(ctx, begin)
	}
	defer func() {
		if err != nil && sh != nil {
			// Only handle end stats if err != nil.
			end := &stats.End{
				Client: true,
				Error:  err,
			}
			sh.HandleRPC(ctx, end)
		}
	}()
	gopts := BalancerGetOptions{
		BlockingWait: !c.failFast,
	}
	for {
		t, put, err = cc.getTransport(ctx, gopts)
		if err != nil {
			// TODO(zhaoq): Probably revisit the error handling.
			if _, ok := status.FromError(err); ok {
				return nil, err
			}
			if err == errConnClosing || err == errConnUnavailable {
				if c.failFast {
					return nil, Errorf(codes.Unavailable, "%v", err)
				}
				continue
			}
			// All the other errors are treated as Internal errors.
			return nil, Errorf(codes.Internal, "%v", err)
		}

		s, err = t.NewStream(ctx, callHdr)
		if err != nil {
			if _, ok := err.(transport.ConnectionError); ok && put != nil {
				// If error is connection error, transport was sending data on wire,
				// and we are not sure if anything has been sent on wire.
				// If error is not connection error, we are sure nothing has been sent.
				updateRPCInfoInContext(ctx, rpcInfo{bytesSent: true, bytesReceived: false})
			}
			if put != nil {
				put()
				put = nil
			}
			if _, ok := err.(transport.ConnectionError); (ok || err == transport.ErrStreamDrain) && !c.failFast {
				continue
			}
			return nil, toRPCErr(err)
		}
		break
	}
	cs := &clientStream{
		opts:       opts,
		c:          c,
		desc:       desc,
		codec:      cc.dopts.codec,
		cp:         cc.dopts.cp,
		dc:         cc.dopts.dc,
		maxMsgSize: cc.dopts.maxMsgSize,
		cancel:     cancel,

		put: put,
		t:   t,
		s:   s,
		p:   &parser{r: s},

		tracing: EnableTracing,
		trInfo:  trInfo,

		statsCtx:     ctx,
		statsHandler: cc.dopts.copts.StatsHandler,
	}
	if cc.dopts.cp != nil {
		cs.cbuf = new(bytes.Buffer)
	}
	// Listen on ctx.Done() to detect cancellation and s.Done() to detect normal termination
	// when there is no pending I/O operations on this stream.
	go func() {
		select {
		case <-t.Error():
			// Incur transport error, simply exit.
		case <-cc.ctx.Done():
			cs.finish(ErrClientConnClosing)
			cs.closeTransportStream(ErrClientConnClosing)
		case <-s.Done():
			// TODO: The trace of the RPC is terminated here when there is no pending
			// I/O, which is probably not the optimal solution.
			cs.finish(s.Status().Err())
			cs.closeTransportStream(nil)
		case <-s.GoAway():
			cs.finish(errConnDrain)
			cs.closeTransportStream(errConnDrain)
		case <-s.Context().Done():
			err := s.Context().Err()
			cs.finish(err)
			cs.closeTransportStream(transport.ContextErr(err))
		}
	}()
	return cs, nil
}

// clientStream implements a client side Stream.
type clientStream struct {
	opts       []CallOption
	c          callInfo
	t          transport.ClientTransport
	s          *transport.Stream
	p          *parser
	desc       *StreamDesc
	codec      Codec
	cp         Compressor
	cbuf       *bytes.Buffer
	dc         Decompressor
	maxMsgSize int
	cancel     context.CancelFunc

	tracing bool // set to EnableTracing when the clientStream is created.

	mu       sync.Mutex
	put      func()
	closed   bool
	finished bool
	// trInfo.tr is set when the clientStream is created (if EnableTracing is true),
	// and is set to nil when the clientStream's finish method is called.
	trInfo traceInfo

	// statsCtx keeps the user context for stats handling.
	// All stats collection should use the statsCtx (instead of the stream context)
	// so that all the generated stats for a particular RPC can be associated in the processing phase.
	statsCtx     context.Context
	statsHandler stats.Handler
}

func (cs *clientStream) Context() context.Context {
	return cs.s.Context()
}

func (cs *clientStream) Header() (metadata.MD, error) {
	m, err := cs.s.Header()
	if err != nil {
		if _, ok := err.(transport.ConnectionError); !ok {
			cs.closeTransportStream(err)
		}
	}
	return m, err
}

func (cs *clientStream) Trailer() metadata.MD {
	return cs.s.Trailer()
}

func (cs *clientStream) SendMsg(m interface{}) (err error) {
	if cs.tracing {
		cs.mu.Lock()
		if cs.trInfo.tr != nil {
			cs.trInfo.tr.LazyLog(&payload{sent: true, msg: m}, true)
		}
		cs.mu.Unlock()
	}
	// TODO Investigate how to signal the stats handling party.
	// generate error stats if err != nil && err != io.EOF?
	defer func() {
		if err != nil {
			cs.finish(err)
		}
		if err == nil {
			return
		}
		if err == io.EOF {
			// Specialize the process for server streaming. SendMesg is only called
			// once when creating the stream object. io.EOF needs to be skipped when
			// the rpc is early finished (before the stream object is created.).
			// TODO: It is probably better to move this into the generated code.
			if !cs.desc.ClientStreams && cs.desc.ServerStreams {
				err = nil
			}
			return
		}
		if _, ok := err.(transport.ConnectionError); !ok {
			cs.closeTransportStream(err)
		}
		err = toRPCErr(err)
	}()
	var outPayload *stats.OutPayload
	if cs.statsHandler != nil {
		outPayload = &stats.OutPayload{
			Client: true,
		}
	}
	out, err := encode(cs.codec, m, cs.cp, cs.cbuf, outPayload)
	defer func() {
		if cs.cbuf != nil {
			cs.cbuf.Reset()
		}
	}()
	if err != nil {
		return Errorf(codes.Internal, "grpc: %v", err)
	}
	err = cs.t.Write(cs.s, out, &transport.Options{Last: false})
	if err == nil && outPayload != nil {
		outPayload.SentTime = time.Now()
		cs.statsHandler.HandleRPC(cs.statsCtx, outPayload)
	}
	return err
}

func (cs *clientStream) RecvMsg(m interface{}) (err error) {
	var inPayload *stats.InPayload
	if cs.statsHandler != nil {
		inPayload = &stats.InPayload{
			Client: true,
		}
	}
	err = recv(cs.p, cs.codec, cs.s, cs.dc, m, cs.maxMsgSize, inPayload)
	defer func() {
		// err != nil indicates the termination of the stream.
		if err != nil {
			cs.finish(err)
		}
	}()
	if err == nil {
		if cs.tracing {
			cs.mu.Lock()
			if cs.trInfo.tr != nil {
				cs.trInfo.tr.LazyLog(&payload{sent: false, msg: m}, true)
			}
			cs.mu.Unlock()
		}
		if inPayload != nil {
			cs.statsHandler.HandleRPC(cs.statsCtx, inPayload)
		}
		if !cs.desc.ClientStreams || cs.desc.ServerStreams {
			return
		}
		// Special handling for client streaming rpc.
		// This recv expects EOF or errors, so we don't collect inPayload.
		err = recv(cs.p, cs.codec, cs.s, cs.dc, m, cs.maxMsgSize, nil)
		cs.closeTransportStream(err)
		if err == nil {
			return toRPCErr(errors.New("grpc: client streaming protocol violation: get <nil>, want <EOF>"))
		}
		if err == io.EOF {
			if se := cs.s.Status().Err(); se != nil {
				return se
			}
			cs.finish(err)
			return nil
		}
		return toRPCErr(err)
	}
	if _, ok := err.(transport.ConnectionError); !ok {
		cs.closeTransportStream(err)
	}
	if err == io.EOF {
		if statusErr := cs.s.Status().Err(); statusErr != nil {
			return statusErr
		}
		// Returns io.EOF to indicate the end of the stream.
		return
	}
	return toRPCErr(err)
}

func (cs *clientStream) CloseSend() (err error) {
	err = cs.t.Write(cs.s, nil, &transport.Options{Last: true})
	defer func() {
		if err != nil {
			cs.finish(err)
		}
	}()
	if err == nil || err == io.EOF {
		return nil
	}
	if _, ok := err.(transport.ConnectionError); !ok {
		cs.closeTransportStream(err)
	}
	err = toRPCErr(err)
	return
}

func (cs *clientStream) closeTransportStream(err error) {
	cs.mu.Lock()
	if cs.closed {
		cs.mu.Unlock()
		return
	}
	cs.closed = true
	cs.mu.Unlock()
	cs.t.CloseStream(cs.s, err)
}

func (cs *clientStream) finish(err error) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	if cs.finished {
		return
	}
	cs.finished = true
	defer func() {
		if cs.cancel != nil {
			cs.cancel()
		}
	}()
	for _, o := range cs.opts {
		o.after(&cs.c)
	}
	if cs.put != nil {
		updateRPCInfoInContext(cs.s.Context(), rpcInfo{
			bytesSent:     cs.s.BytesSent(),
			bytesReceived: cs.s.BytesReceived(),
		})
		cs.put()
		cs.put = nil
	}
	if cs.statsHandler != nil {
		end := &stats.End{
			Client:  true,
			EndTime: time.Now(),
		}
		if err != io.EOF {
			// end.Error is nil if the RPC finished successfully.
			end.Error = toRPCErr(err)
		}
		cs.statsHandler.HandleRPC(cs.statsCtx, end)
	}
	if !cs.tracing {
		return
	}
	if cs.trInfo.tr != nil {
		if err == nil || err == io.EOF {
			cs.trInfo.tr.LazyPrintf("RPC: [OK]")
		} else {
			cs.trInfo.tr.LazyPrintf("RPC: [%v]", err)
			cs.trInfo.tr.SetError()
		}
		cs.trInfo.tr.Finish()
		cs.trInfo.tr = nil
	}
}

// ServerStream defines the interface a server stream has to satisfy.
type ServerStream interface {
	// SetHeader sets the header metadata. It may be called multiple times.
	// When call multiple times, all the provided metadata will be merged.
	// All the metadata will be sent out when one of the following happens:
	//  - ServerStream.SendHeader() is called;
	//  - The first response is sent out;
	//  - An RPC status is sent out (error or success).
	SetHeader(metadata.MD) error
	// SendHeader sends the header metadata.
	// The provided md and headers set by SetHeader() will be sent.
	// It fails if called multiple times.
	SendHeader(metadata.MD) error
	// SetTrailer sets the trailer metadata which will be sent with the RPC status.
	// When called more than once, all the provided metadata will be merged.
	SetTrailer(metadata.MD)
	Stream
}

// serverStream implements a server side Stream.
type serverStream struct {
	t          transport.ServerTransport
	s          *transport.Stream
	p          *parser
	codec      Codec
	cp         Compressor
	dc         Decompressor
	cbuf       *bytes.Buffer
	maxMsgSize int
	trInfo     *traceInfo

	statsHandler stats.Handler

	mu sync.Mutex // protects trInfo.tr after the service handler runs.
}

func (ss *serverStream) Context() context.Context {
	return ss.s.Context()
}

func (ss *serverStream) SetHeader(md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	return ss.s.SetHeader(md)
}

func (ss *serverStream) SendHeader(md metadata.MD) error {
	return ss.t.WriteHeader(ss.s, md)
}

func (ss *serverStream) SetTrailer(md metadata.MD) {
	if md.Len() == 0 {
		return
	}
	ss.s.SetTrailer(md)
	return
}

func (ss *serverStream) SendMsg(m interface{}) (err error) {
	defer func() {
		if ss.trInfo != nil {
			ss.mu.Lock()
			if ss.trInfo.tr != nil {
				if err == nil {
					ss.trInfo.tr.LazyLog(&payload{sent: true, msg: m}, true)
				} else {
					ss.trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
					ss.trInfo.tr.SetError()
				}
			}
			ss.mu.Unlock()
		}
	}()
	var outPayload *stats.OutPayload
	if ss.statsHandler != nil {
		outPayload = &stats.OutPayload{}
	}
	out, err := encode(ss.codec, m, ss.cp, ss.cbuf, outPayload)
	defer func() {
		if ss.cbuf != nil {
			ss.cbuf.Reset()
		}
	}()
	if err != nil {
		err = Errorf(codes.Internal, "grpc: %v", err)
		return err
	}
	if err := ss.t.Write(ss.s, out, &transport.Options{Last: false}); err != nil {
		return toRPCErr(err)
	}
	if outPayload != nil {
		outPayload.SentTime = time.Now()
		ss.statsHandler.HandleRPC(ss.s.Context(), outPayload)
	}
	return nil
}

func (ss *serverStream) RecvMsg(m interface{}) (err error) {
	defer func() {
		if ss.trInfo != nil {
			ss.mu.Lock()
			if ss.trInfo.tr != nil {
				if err == nil {
					ss.trInfo.tr.LazyLog(&payload{sent: false, msg: m}, true)
				} else if err != io.EOF {
					ss.trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
					ss.trInfo.tr.SetError()
				}
			}
			ss.mu.Unlock()
		}
	}()
	var inPayload *stats.InPayload
	if ss.statsHandler != nil {
		inPayload = &stats.InPayload{}
	}
	if err := recv(ss.p, ss.codec, ss.s, ss.dc, m, ss.maxMsgSize, inPayload); err != nil {
		if err == io.EOF {
			return err
		}
		if err == io.ErrUnexpectedEOF {
			err = Errorf(codes.Internal, io.ErrUnexpectedEOF.Error())
		}
		return toRPCErr(err)
	}
	if inPayload != nil {
		ss.statsHandler.HandleRPC(ss.s.Context(), inPayload)
	}
	return nil
}
