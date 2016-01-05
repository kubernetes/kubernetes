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
	"io"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/transport"
)

// recvResponse receives and parses an RPC response.
// On error, it returns the error and indicates whether the call should be retried.
//
// TODO(zhaoq): Check whether the received message sequence is valid.
func recvResponse(codec Codec, t transport.ClientTransport, c *callInfo, stream *transport.Stream, reply interface{}) error {
	// Try to acquire header metadata from the server if there is any.
	var err error
	c.headerMD, err = stream.Header()
	if err != nil {
		return err
	}
	p := &parser{s: stream}
	for {
		if err = recv(p, codec, reply); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}
	c.trailerMD = stream.Trailer()
	return nil
}

// sendRequest writes out various information of an RPC such as Context and Message.
func sendRequest(ctx context.Context, codec Codec, callHdr *transport.CallHdr, t transport.ClientTransport, args interface{}, opts *transport.Options) (_ *transport.Stream, err error) {
	stream, err := t.NewStream(ctx, callHdr)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			if _, ok := err.(transport.ConnectionError); !ok {
				t.CloseStream(stream, err)
			}
		}
	}()
	// TODO(zhaoq): Support compression.
	outBuf, err := encode(codec, args, compressionNone)
	if err != nil {
		return nil, transport.StreamErrorf(codes.Internal, "grpc: %v", err)
	}
	err = t.Write(stream, outBuf, opts)
	if err != nil {
		return nil, err
	}
	// Sent successfully.
	return stream, nil
}

// callInfo contains all related configuration and information about an RPC.
type callInfo struct {
	failFast  bool
	headerMD  metadata.MD
	trailerMD metadata.MD
	traceInfo traceInfo // in trace.go
}

// Invoke is called by the generated code. It sends the RPC request on the
// wire and returns after response is received.
func Invoke(ctx context.Context, method string, args, reply interface{}, cc *ClientConn, opts ...CallOption) (err error) {
	var c callInfo
	for _, o := range opts {
		if err := o.before(&c); err != nil {
			return toRPCErr(err)
		}
	}
	defer func() {
		for _, o := range opts {
			o.after(&c)
		}
	}()

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
			if err != nil {
				c.traceInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
				c.traceInfo.tr.SetError()
			}
		}()
	}
	callHdr := &transport.CallHdr{
		Host:   cc.authority,
		Method: method,
	}
	topts := &transport.Options{
		Last:  true,
		Delay: false,
	}
	var (
		ts      int   // track the transport sequence number
		lastErr error // record the error that happened
	)
	for {
		var (
			err    error
			t      transport.ClientTransport
			stream *transport.Stream
		)
		// TODO(zhaoq): Need a formal spec of retry strategy for non-failfast rpcs.
		if lastErr != nil && c.failFast {
			return toRPCErr(lastErr)
		}
		t, ts, err = cc.wait(ctx, ts)
		if err != nil {
			if lastErr != nil {
				// This was a retry; return the error from the last attempt.
				return toRPCErr(lastErr)
			}
			return toRPCErr(err)
		}
		if c.traceInfo.tr != nil {
			c.traceInfo.tr.LazyLog(&payload{sent: true, msg: args}, true)
		}
		stream, err = sendRequest(ctx, cc.dopts.codec, callHdr, t, args, topts)
		if err != nil {
			if _, ok := err.(transport.ConnectionError); ok {
				lastErr = err
				continue
			}
			if lastErr != nil {
				return toRPCErr(lastErr)
			}
			return toRPCErr(err)
		}
		// Receive the response
		lastErr = recvResponse(cc.dopts.codec, t, &c, stream, reply)
		if _, ok := lastErr.(transport.ConnectionError); ok {
			continue
		}
		if c.traceInfo.tr != nil {
			c.traceInfo.tr.LazyLog(&payload{sent: false, msg: reply}, true)
		}
		t.CloseStream(stream, lastErr)
		if lastErr != nil {
			return toRPCErr(lastErr)
		}
		return Errorf(stream.StatusCode(), stream.StatusDesc())
	}
}
