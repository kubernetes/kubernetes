/*
 * Copyright 2016, Google Inc.
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

// This file is the implementation of a gRPC server using HTTP/2 which
// uses the standard Go http2 Server implementation (via the
// http.Handler interface), rather than speaking low-level HTTP/2
// frames itself. It is the implementation of *grpc.Server.ServeHTTP.

package transport

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
)

// NewServerHandlerTransport returns a ServerTransport handling gRPC
// from inside an http.Handler. It requires that the http Server
// supports HTTP/2.
func NewServerHandlerTransport(w http.ResponseWriter, r *http.Request) (ServerTransport, error) {
	if r.ProtoMajor != 2 {
		return nil, errors.New("gRPC requires HTTP/2")
	}
	if r.Method != "POST" {
		return nil, errors.New("invalid gRPC request method")
	}
	if !validContentType(r.Header.Get("Content-Type")) {
		return nil, errors.New("invalid gRPC request content-type")
	}
	if _, ok := w.(http.Flusher); !ok {
		return nil, errors.New("gRPC requires a ResponseWriter supporting http.Flusher")
	}
	if _, ok := w.(http.CloseNotifier); !ok {
		return nil, errors.New("gRPC requires a ResponseWriter supporting http.CloseNotifier")
	}

	st := &serverHandlerTransport{
		rw:       w,
		req:      r,
		closedCh: make(chan struct{}),
		writes:   make(chan func()),
	}

	if v := r.Header.Get("grpc-timeout"); v != "" {
		to, err := decodeTimeout(v)
		if err != nil {
			return nil, streamErrorf(codes.Internal, "malformed time-out: %v", err)
		}
		st.timeoutSet = true
		st.timeout = to
	}

	var metakv []string
	if r.Host != "" {
		metakv = append(metakv, ":authority", r.Host)
	}
	for k, vv := range r.Header {
		k = strings.ToLower(k)
		if isReservedHeader(k) && !isWhitelistedPseudoHeader(k) {
			continue
		}
		for _, v := range vv {
			if k == "user-agent" {
				// user-agent is special. Copying logic of http_util.go.
				if i := strings.LastIndex(v, " "); i == -1 {
					// There is no application user agent string being set
					continue
				} else {
					v = v[:i]
				}
			}
			metakv = append(metakv, k, v)
		}
	}
	st.headerMD = metadata.Pairs(metakv...)

	return st, nil
}

// serverHandlerTransport is an implementation of ServerTransport
// which replies to exactly one gRPC request (exactly one HTTP request),
// using the net/http.Handler interface. This http.Handler is guaranteed
// at this point to be speaking over HTTP/2, so it's able to speak valid
// gRPC.
type serverHandlerTransport struct {
	rw               http.ResponseWriter
	req              *http.Request
	timeoutSet       bool
	timeout          time.Duration
	didCommonHeaders bool

	headerMD metadata.MD

	closeOnce sync.Once
	closedCh  chan struct{} // closed on Close

	// writes is a channel of code to run serialized in the
	// ServeHTTP (HandleStreams) goroutine. The channel is closed
	// when WriteStatus is called.
	writes chan func()
}

func (ht *serverHandlerTransport) Close() error {
	ht.closeOnce.Do(ht.closeCloseChanOnce)
	return nil
}

func (ht *serverHandlerTransport) closeCloseChanOnce() { close(ht.closedCh) }

func (ht *serverHandlerTransport) RemoteAddr() net.Addr { return strAddr(ht.req.RemoteAddr) }

// strAddr is a net.Addr backed by either a TCP "ip:port" string, or
// the empty string if unknown.
type strAddr string

func (a strAddr) Network() string {
	if a != "" {
		// Per the documentation on net/http.Request.RemoteAddr, if this is
		// set, it's set to the IP:port of the peer (hence, TCP):
		// https://golang.org/pkg/net/http/#Request
		//
		// If we want to support Unix sockets later, we can
		// add our own grpc-specific convention within the
		// grpc codebase to set RemoteAddr to a different
		// format, or probably better: we can attach it to the
		// context and use that from serverHandlerTransport.RemoteAddr.
		return "tcp"
	}
	return ""
}

func (a strAddr) String() string { return string(a) }

// do runs fn in the ServeHTTP goroutine.
func (ht *serverHandlerTransport) do(fn func()) error {
	select {
	case ht.writes <- fn:
		return nil
	case <-ht.closedCh:
		return ErrConnClosing
	}
}

func (ht *serverHandlerTransport) WriteStatus(s *Stream, statusCode codes.Code, statusDesc string) error {
	err := ht.do(func() {
		ht.writeCommonHeaders(s)

		// And flush, in case no header or body has been sent yet.
		// This forces a separation of headers and trailers if this is the
		// first call (for example, in end2end tests's TestNoService).
		ht.rw.(http.Flusher).Flush()

		h := ht.rw.Header()
		h.Set("Grpc-Status", fmt.Sprintf("%d", statusCode))
		if statusDesc != "" {
			h.Set("Grpc-Message", encodeGrpcMessage(statusDesc))
		}
		if md := s.Trailer(); len(md) > 0 {
			for k, vv := range md {
				// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
				if isReservedHeader(k) {
					continue
				}
				for _, v := range vv {
					// http2 ResponseWriter mechanism to
					// send undeclared Trailers after the
					// headers have possibly been written.
					h.Add(http2.TrailerPrefix+k, v)
				}
			}
		}
	})
	close(ht.writes)
	return err
}

// writeCommonHeaders sets common headers on the first write
// call (Write, WriteHeader, or WriteStatus).
func (ht *serverHandlerTransport) writeCommonHeaders(s *Stream) {
	if ht.didCommonHeaders {
		return
	}
	ht.didCommonHeaders = true

	h := ht.rw.Header()
	h["Date"] = nil // suppress Date to make tests happy; TODO: restore
	h.Set("Content-Type", "application/grpc")

	// Predeclare trailers we'll set later in WriteStatus (after the body).
	// This is a SHOULD in the HTTP RFC, and the way you add (known)
	// Trailers per the net/http.ResponseWriter contract.
	// See https://golang.org/pkg/net/http/#ResponseWriter
	// and https://golang.org/pkg/net/http/#example_ResponseWriter_trailers
	h.Add("Trailer", "Grpc-Status")
	h.Add("Trailer", "Grpc-Message")

	if s.sendCompress != "" {
		h.Set("Grpc-Encoding", s.sendCompress)
	}
}

func (ht *serverHandlerTransport) Write(s *Stream, data []byte, opts *Options) error {
	return ht.do(func() {
		ht.writeCommonHeaders(s)
		ht.rw.Write(data)
		if !opts.Delay {
			ht.rw.(http.Flusher).Flush()
		}
	})
}

func (ht *serverHandlerTransport) WriteHeader(s *Stream, md metadata.MD) error {
	return ht.do(func() {
		ht.writeCommonHeaders(s)
		h := ht.rw.Header()
		for k, vv := range md {
			// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
			if isReservedHeader(k) {
				continue
			}
			for _, v := range vv {
				h.Add(k, v)
			}
		}
		ht.rw.WriteHeader(200)
		ht.rw.(http.Flusher).Flush()
	})
}

func (ht *serverHandlerTransport) HandleStreams(startStream func(*Stream), traceCtx func(context.Context, string) context.Context) {
	// With this transport type there will be exactly 1 stream: this HTTP request.

	var ctx context.Context
	var cancel context.CancelFunc
	if ht.timeoutSet {
		ctx, cancel = context.WithTimeout(context.Background(), ht.timeout)
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}

	// requestOver is closed when either the request's context is done
	// or the status has been written via WriteStatus.
	requestOver := make(chan struct{})

	// clientGone receives a single value if peer is gone, either
	// because the underlying connection is dead or because the
	// peer sends an http2 RST_STREAM.
	clientGone := ht.rw.(http.CloseNotifier).CloseNotify()
	go func() {
		select {
		case <-requestOver:
			return
		case <-ht.closedCh:
		case <-clientGone:
		}
		cancel()
	}()

	req := ht.req

	s := &Stream{
		id:            0,            // irrelevant
		windowHandler: func(int) {}, // nothing
		cancel:        cancel,
		buf:           newRecvBuffer(),
		st:            ht,
		method:        req.URL.Path,
		recvCompress:  req.Header.Get("grpc-encoding"),
	}
	pr := &peer.Peer{
		Addr: ht.RemoteAddr(),
	}
	if req.TLS != nil {
		pr.AuthInfo = credentials.TLSInfo{State: *req.TLS}
	}
	ctx = metadata.NewContext(ctx, ht.headerMD)
	ctx = peer.NewContext(ctx, pr)
	s.ctx = newContextWithStream(ctx, s)
	s.dec = &recvBufferReader{ctx: s.ctx, recv: s.buf}

	// readerDone is closed when the Body.Read-ing goroutine exits.
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)

		// TODO: minimize garbage, optimize recvBuffer code/ownership
		const readSize = 8196
		for buf := make([]byte, readSize); ; {
			n, err := req.Body.Read(buf)
			if n > 0 {
				s.buf.put(&recvMsg{data: buf[:n:n]})
				buf = buf[n:]
			}
			if err != nil {
				s.buf.put(&recvMsg{err: mapRecvMsgError(err)})
				return
			}
			if len(buf) == 0 {
				buf = make([]byte, readSize)
			}
		}
	}()

	// startStream is provided by the *grpc.Server's serveStreams.
	// It starts a goroutine serving s and exits immediately.
	// The goroutine that is started is the one that then calls
	// into ht, calling WriteHeader, Write, WriteStatus, Close, etc.
	startStream(s)

	ht.runStream()
	close(requestOver)

	// Wait for reading goroutine to finish.
	req.Body.Close()
	<-readerDone
}

func (ht *serverHandlerTransport) runStream() {
	for {
		select {
		case fn, ok := <-ht.writes:
			if !ok {
				return
			}
			fn()
		case <-ht.closedCh:
			return
		}
	}
}

func (ht *serverHandlerTransport) Drain() {
	panic("Drain() is not implemented")
}

// mapRecvMsgError returns the non-nil err into the appropriate
// error value as expected by callers of *grpc.parser.recvMsg.
// In particular, in can only be:
//   * io.EOF
//   * io.ErrUnexpectedEOF
//   * of type transport.ConnectionError
//   * of type transport.StreamError
func mapRecvMsgError(err error) error {
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return err
	}
	if se, ok := err.(http2.StreamError); ok {
		if code, ok := http2ErrConvTab[se.Code]; ok {
			return StreamError{
				Code: code,
				Desc: se.Error(),
			}
		}
	}
	return connectionErrorf(true, err, err.Error())
}
