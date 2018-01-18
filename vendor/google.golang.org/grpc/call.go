/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"bytes"
	"io"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

// recvResponse receives and parses an RPC response.
// On error, it returns the error and indicates whether the call should be retried.
//
// TODO(zhaoq): Check whether the received message sequence is valid.
// TODO ctx is used for stats collection and processing. It is the context passed from the application.
func recvResponse(ctx context.Context, dopts dialOptions, t transport.ClientTransport, c *callInfo, stream *transport.Stream, reply interface{}) (err error) {
	// Try to acquire header metadata from the server if there is any.
	defer func() {
		if err != nil {
			if _, ok := err.(transport.ConnectionError); !ok {
				t.CloseStream(stream, err)
			}
		}
	}()
	c.headerMD, err = stream.Header()
	if err != nil {
		return
	}
	p := &parser{r: stream}
	var inPayload *stats.InPayload
	if dopts.copts.StatsHandler != nil {
		inPayload = &stats.InPayload{
			Client: true,
		}
	}
	for {
		if c.maxReceiveMessageSize == nil {
			return Errorf(codes.Internal, "callInfo maxReceiveMessageSize field uninitialized(nil)")
		}
		if err = recv(p, dopts.codec, stream, dopts.dc, reply, *c.maxReceiveMessageSize, inPayload); err != nil {
			if err == io.EOF {
				break
			}
			return
		}
	}
	if inPayload != nil && err == io.EOF && stream.Status().Code() == codes.OK {
		// TODO in the current implementation, inTrailer may be handled before inPayload in some cases.
		// Fix the order if necessary.
		dopts.copts.StatsHandler.HandleRPC(ctx, inPayload)
	}
	c.trailerMD = stream.Trailer()
	return nil
}

// sendRequest writes out various information of an RPC such as Context and Message.
func sendRequest(ctx context.Context, dopts dialOptions, compressor Compressor, c *callInfo, callHdr *transport.CallHdr, stream *transport.Stream, t transport.ClientTransport, args interface{}, opts *transport.Options) (err error) {
	defer func() {
		if err != nil {
			// If err is connection error, t will be closed, no need to close stream here.
			if _, ok := err.(transport.ConnectionError); !ok {
				t.CloseStream(stream, err)
			}
		}
	}()
	var (
		cbuf       *bytes.Buffer
		outPayload *stats.OutPayload
	)
	if compressor != nil {
		cbuf = new(bytes.Buffer)
	}
	if dopts.copts.StatsHandler != nil {
		outPayload = &stats.OutPayload{
			Client: true,
		}
	}
	hdr, data, err := encode(dopts.codec, args, compressor, cbuf, outPayload)
	if err != nil {
		return err
	}
	if c.maxSendMessageSize == nil {
		return Errorf(codes.Internal, "callInfo maxSendMessageSize field uninitialized(nil)")
	}
	if len(data) > *c.maxSendMessageSize {
		return Errorf(codes.ResourceExhausted, "grpc: trying to send message larger than max (%d vs. %d)", len(data), *c.maxSendMessageSize)
	}
	err = t.Write(stream, hdr, data, opts)
	if err == nil && outPayload != nil {
		outPayload.SentTime = time.Now()
		dopts.copts.StatsHandler.HandleRPC(ctx, outPayload)
	}
	// t.NewStream(...) could lead to an early rejection of the RPC (e.g., the service/method
	// does not exist.) so that t.Write could get io.EOF from wait(...). Leave the following
	// recvResponse to get the final status.
	if err != nil && err != io.EOF {
		return err
	}
	// Sent successfully.
	return nil
}

// Invoke sends the RPC request on the wire and returns after response is received.
// Invoke is called by generated code. Also users can call Invoke directly when it
// is really needed in their use cases.
func Invoke(ctx context.Context, method string, args, reply interface{}, cc *ClientConn, opts ...CallOption) error {
	if cc.dopts.unaryInt != nil {
		return cc.dopts.unaryInt(ctx, method, args, reply, cc, invoke, opts...)
	}
	return invoke(ctx, method, args, reply, cc, opts...)
}

func invoke(ctx context.Context, method string, args, reply interface{}, cc *ClientConn, opts ...CallOption) (e error) {
	c := defaultCallInfo()
	mc := cc.GetMethodConfig(method)
	if mc.WaitForReady != nil {
		c.failFast = !*mc.WaitForReady
	}

	if mc.Timeout != nil && *mc.Timeout >= 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *mc.Timeout)
		defer cancel()
	}

	opts = append(cc.dopts.callOptions, opts...)
	for _, o := range opts {
		if err := o.before(c); err != nil {
			return toRPCErr(err)
		}
	}
	defer func() {
		for _, o := range opts {
			o.after(c)
		}
	}()

	c.maxSendMessageSize = getMaxSize(mc.MaxReqSize, c.maxSendMessageSize, defaultClientMaxSendMessageSize)
	c.maxReceiveMessageSize = getMaxSize(mc.MaxRespSize, c.maxReceiveMessageSize, defaultClientMaxReceiveMessageSize)

	if EnableTracing {
		c.traceInfo.tr = trace.New("grpc.Sent."+methodFamily(method), method)
		defer c.traceInfo.tr.Finish()
		c.traceInfo.firstLine.client = true
		if deadline, ok := ctx.Deadline(); ok {
			c.traceInfo.firstLine.deadline = deadline.Sub(time.Now())
		}
		c.traceInfo.tr.LazyLog(&c.traceInfo.firstLine, false)
		// TODO(dsymonds): Arrange for c.traceInfo.firstLine.remoteAddr to be set.
		defer func() {
			if e != nil {
				c.traceInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{e}}, true)
				c.traceInfo.tr.SetError()
			}
		}()
	}
	ctx = newContextWithRPCInfo(ctx, c.failFast)
	sh := cc.dopts.copts.StatsHandler
	if sh != nil {
		ctx = sh.TagRPC(ctx, &stats.RPCTagInfo{FullMethodName: method, FailFast: c.failFast})
		begin := &stats.Begin{
			Client:    true,
			BeginTime: time.Now(),
			FailFast:  c.failFast,
		}
		sh.HandleRPC(ctx, begin)
		defer func() {
			end := &stats.End{
				Client:  true,
				EndTime: time.Now(),
				Error:   e,
			}
			sh.HandleRPC(ctx, end)
		}()
	}
	topts := &transport.Options{
		Last:  true,
		Delay: false,
	}
	for {
		var (
			err    error
			t      transport.ClientTransport
			stream *transport.Stream
			// Record the done handler from Balancer.Get(...). It is called once the
			// RPC has completed or failed.
			done func(balancer.DoneInfo)
		)
		// TODO(zhaoq): Need a formal spec of fail-fast.
		callHdr := &transport.CallHdr{
			Host:   cc.authority,
			Method: method,
		}
		if cc.dopts.cp != nil {
			callHdr.SendCompress = cc.dopts.cp.Type()
		}
		if c.creds != nil {
			callHdr.Creds = c.creds
		}

		t, done, err = cc.getTransport(ctx, c.failFast)
		if err != nil {
			// TODO(zhaoq): Probably revisit the error handling.
			if _, ok := status.FromError(err); ok {
				return err
			}
			if err == errConnClosing || err == errConnUnavailable {
				if c.failFast {
					return Errorf(codes.Unavailable, "%v", err)
				}
				continue
			}
			// All the other errors are treated as Internal errors.
			return Errorf(codes.Internal, "%v", err)
		}
		if c.traceInfo.tr != nil {
			c.traceInfo.tr.LazyLog(&payload{sent: true, msg: args}, true)
		}
		stream, err = t.NewStream(ctx, callHdr)
		if err != nil {
			if done != nil {
				if _, ok := err.(transport.ConnectionError); ok {
					// If error is connection error, transport was sending data on wire,
					// and we are not sure if anything has been sent on wire.
					// If error is not connection error, we are sure nothing has been sent.
					updateRPCInfoInContext(ctx, rpcInfo{bytesSent: true, bytesReceived: false})
				}
				done(balancer.DoneInfo{Err: err})
			}
			if _, ok := err.(transport.ConnectionError); (ok || err == transport.ErrStreamDrain) && !c.failFast {
				continue
			}
			return toRPCErr(err)
		}
		if peer, ok := peer.FromContext(stream.Context()); ok {
			c.peer = peer
		}
		err = sendRequest(ctx, cc.dopts, cc.dopts.cp, c, callHdr, stream, t, args, topts)
		if err != nil {
			if done != nil {
				updateRPCInfoInContext(ctx, rpcInfo{
					bytesSent:     stream.BytesSent(),
					bytesReceived: stream.BytesReceived(),
				})
				done(balancer.DoneInfo{Err: err})
			}
			// Retry a non-failfast RPC when
			// i) there is a connection error; or
			// ii) the server started to drain before this RPC was initiated.
			if _, ok := err.(transport.ConnectionError); (ok || err == transport.ErrStreamDrain) && !c.failFast {
				continue
			}
			return toRPCErr(err)
		}
		err = recvResponse(ctx, cc.dopts, t, c, stream, reply)
		if err != nil {
			if done != nil {
				updateRPCInfoInContext(ctx, rpcInfo{
					bytesSent:     stream.BytesSent(),
					bytesReceived: stream.BytesReceived(),
				})
				done(balancer.DoneInfo{Err: err})
			}
			if _, ok := err.(transport.ConnectionError); (ok || err == transport.ErrStreamDrain) && !c.failFast {
				continue
			}
			return toRPCErr(err)
		}
		if c.traceInfo.tr != nil {
			c.traceInfo.tr.LazyLog(&payload{sent: false, msg: reply}, true)
		}
		t.CloseStream(stream, nil)
		if done != nil {
			updateRPCInfoInContext(ctx, rpcInfo{
				bytesSent:     stream.BytesSent(),
				bytesReceived: stream.BytesReceived(),
			})
			done(balancer.DoneInfo{Err: err})
		}
		return stream.Status().Err()
	}
}
