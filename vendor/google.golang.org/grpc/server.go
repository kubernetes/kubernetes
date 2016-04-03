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
	"fmt"
	"io"
	"net"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/transport"
)

type methodHandler func(srv interface{}, ctx context.Context, dec func(interface{}) error) (interface{}, error)

// MethodDesc represents an RPC service's method specification.
type MethodDesc struct {
	MethodName string
	Handler    methodHandler
}

// ServiceDesc represents an RPC service's specification.
type ServiceDesc struct {
	ServiceName string
	// The pointer to the service interface. Used to check whether the user
	// provided implementation satisfies the interface requirements.
	HandlerType interface{}
	Methods     []MethodDesc
	Streams     []StreamDesc
}

// service consists of the information of the server serving this service and
// the methods in this service.
type service struct {
	server interface{} // the server for service methods
	md     map[string]*MethodDesc
	sd     map[string]*StreamDesc
}

// Server is a gRPC server to serve RPC requests.
type Server struct {
	opts   options
	mu     sync.Mutex
	lis    map[net.Listener]bool
	conns  map[transport.ServerTransport]bool
	m      map[string]*service // service name -> service info
	events trace.EventLog
}

type options struct {
	creds                credentials.Credentials
	codec                Codec
	cp                   Compressor
	dc                   Decompressor
	maxConcurrentStreams uint32
}

// A ServerOption sets options.
type ServerOption func(*options)

// CustomCodec returns a ServerOption that sets a codec for message marshaling and unmarshaling.
func CustomCodec(codec Codec) ServerOption {
	return func(o *options) {
		o.codec = codec
	}
}

func RPCCompressor(cp Compressor) ServerOption {
	return func(o *options) {
		o.cp = cp
	}
}

func RPCDecompressor(dc Decompressor) ServerOption {
	return func(o *options) {
		o.dc = dc
	}
}

// MaxConcurrentStreams returns a ServerOption that will apply a limit on the number
// of concurrent streams to each ServerTransport.
func MaxConcurrentStreams(n uint32) ServerOption {
	return func(o *options) {
		o.maxConcurrentStreams = n
	}
}

// Creds returns a ServerOption that sets credentials for server connections.
func Creds(c credentials.Credentials) ServerOption {
	return func(o *options) {
		o.creds = c
	}
}

// NewServer creates a gRPC server which has no service registered and has not
// started to accept requests yet.
func NewServer(opt ...ServerOption) *Server {
	var opts options
	for _, o := range opt {
		o(&opts)
	}
	if opts.codec == nil {
		// Set the default codec.
		opts.codec = protoCodec{}
	}
	s := &Server{
		lis:   make(map[net.Listener]bool),
		opts:  opts,
		conns: make(map[transport.ServerTransport]bool),
		m:     make(map[string]*service),
	}
	if EnableTracing {
		_, file, line, _ := runtime.Caller(1)
		s.events = trace.NewEventLog("grpc.Server", fmt.Sprintf("%s:%d", file, line))
	}
	return s
}

// printf records an event in s's event log, unless s has been stopped.
// REQUIRES s.mu is held.
func (s *Server) printf(format string, a ...interface{}) {
	if s.events != nil {
		s.events.Printf(format, a...)
	}
}

// errorf records an error in s's event log, unless s has been stopped.
// REQUIRES s.mu is held.
func (s *Server) errorf(format string, a ...interface{}) {
	if s.events != nil {
		s.events.Errorf(format, a...)
	}
}

// RegisterService register a service and its implementation to the gRPC
// server. Called from the IDL generated code. This must be called before
// invoking Serve.
func (s *Server) RegisterService(sd *ServiceDesc, ss interface{}) {
	ht := reflect.TypeOf(sd.HandlerType).Elem()
	st := reflect.TypeOf(ss)
	if !st.Implements(ht) {
		grpclog.Fatalf("grpc: Server.RegisterService found the handler of type %v that does not satisfy %v", st, ht)
	}
	s.register(sd, ss)
}

func (s *Server) register(sd *ServiceDesc, ss interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.printf("RegisterService(%q)", sd.ServiceName)
	if _, ok := s.m[sd.ServiceName]; ok {
		grpclog.Fatalf("grpc: Server.RegisterService found duplicate service registration for %q", sd.ServiceName)
	}
	srv := &service{
		server: ss,
		md:     make(map[string]*MethodDesc),
		sd:     make(map[string]*StreamDesc),
	}
	for i := range sd.Methods {
		d := &sd.Methods[i]
		srv.md[d.MethodName] = d
	}
	for i := range sd.Streams {
		d := &sd.Streams[i]
		srv.sd[d.StreamName] = d
	}
	s.m[sd.ServiceName] = srv
}

var (
	// ErrServerStopped indicates that the operation is now illegal because of
	// the server being stopped.
	ErrServerStopped = errors.New("grpc: the server has been stopped")
)

// Serve accepts incoming connections on the listener lis, creating a new
// ServerTransport and service goroutine for each. The service goroutines
// read gRPC request and then call the registered handlers to reply to them.
// Service returns when lis.Accept fails.
func (s *Server) Serve(lis net.Listener) error {
	s.mu.Lock()
	s.printf("serving")
	if s.lis == nil {
		s.mu.Unlock()
		return ErrServerStopped
	}
	s.lis[lis] = true
	s.mu.Unlock()
	defer func() {
		lis.Close()
		s.mu.Lock()
		delete(s.lis, lis)
		s.mu.Unlock()
	}()
	for {
		c, err := lis.Accept()
		if err != nil {
			s.mu.Lock()
			s.printf("done serving; Accept = %v", err)
			s.mu.Unlock()
			return err
		}
		var authInfo credentials.AuthInfo
		if creds, ok := s.opts.creds.(credentials.TransportAuthenticator); ok {
			var conn net.Conn
			conn, authInfo, err = creds.ServerHandshake(c)
			if err != nil {
				s.mu.Lock()
				s.errorf("ServerHandshake(%q) failed: %v", c.RemoteAddr(), err)
				s.mu.Unlock()
				grpclog.Println("grpc: Server.Serve failed to complete security handshake.")
				continue
			}
			c = conn
		}
		s.mu.Lock()
		if s.conns == nil {
			s.mu.Unlock()
			c.Close()
			return nil
		}
		s.mu.Unlock()

		go s.serveNewHTTP2Transport(c, authInfo)
	}
}

func (s *Server) serveNewHTTP2Transport(c net.Conn, authInfo credentials.AuthInfo) {
	st, err := transport.NewServerTransport("http2", c, s.opts.maxConcurrentStreams, authInfo)
	if err != nil {
		s.mu.Lock()
		s.errorf("NewServerTransport(%q) failed: %v", c.RemoteAddr(), err)
		s.mu.Unlock()
		c.Close()
		grpclog.Println("grpc: Server.Serve failed to create ServerTransport: ", err)
		return
	}
	if !s.addConn(st) {
		c.Close()
		return
	}
	s.serveStreams(st)
}

func (s *Server) serveStreams(st transport.ServerTransport) {
	defer s.removeConn(st)
	defer st.Close()
	var wg sync.WaitGroup
	st.HandleStreams(func(stream *transport.Stream) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.handleStream(st, stream, s.traceInfo(st, stream))
		}()
	})
	wg.Wait()
}

// traceInfo returns a traceInfo and associates it with stream, if tracing is enabled.
// If tracing is not enabled, it returns nil.
func (s *Server) traceInfo(st transport.ServerTransport, stream *transport.Stream) (trInfo *traceInfo) {
	if !EnableTracing {
		return nil
	}
	trInfo = &traceInfo{
		tr: trace.New("grpc.Recv."+methodFamily(stream.Method()), stream.Method()),
	}
	trInfo.firstLine.client = false
	trInfo.firstLine.remoteAddr = st.RemoteAddr()
	stream.TraceContext(trInfo.tr)
	if dl, ok := stream.Context().Deadline(); ok {
		trInfo.firstLine.deadline = dl.Sub(time.Now())
	}
	return trInfo
}

func (s *Server) addConn(st transport.ServerTransport) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conns == nil {
		return false
	}
	s.conns[st] = true
	return true
}

func (s *Server) removeConn(st transport.ServerTransport) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conns != nil {
		delete(s.conns, st)
	}
}

func (s *Server) sendResponse(t transport.ServerTransport, stream *transport.Stream, msg interface{}, cp Compressor, opts *transport.Options) error {
	var cbuf *bytes.Buffer
	if cp != nil {
		cbuf = new(bytes.Buffer)
	}
	p, err := encode(s.opts.codec, msg, cp, cbuf)
	if err != nil {
		// This typically indicates a fatal issue (e.g., memory
		// corruption or hardware faults) the application program
		// cannot handle.
		//
		// TODO(zhaoq): There exist other options also such as only closing the
		// faulty stream locally and remotely (Other streams can keep going). Find
		// the optimal option.
		grpclog.Fatalf("grpc: Server failed to encode response %v", err)
	}
	return t.Write(stream, p, opts)
}

func (s *Server) processUnaryRPC(t transport.ServerTransport, stream *transport.Stream, srv *service, md *MethodDesc, trInfo *traceInfo) (err error) {
	if trInfo != nil {
		defer trInfo.tr.Finish()
		trInfo.firstLine.client = false
		trInfo.tr.LazyLog(&trInfo.firstLine, false)
		defer func() {
			if err != nil && err != io.EOF {
				trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
				trInfo.tr.SetError()
			}
		}()
	}
	p := &parser{s: stream}
	for {
		pf, req, err := p.recvMsg()
		if err == io.EOF {
			// The entire stream is done (for unary RPC only).
			return err
		}
		if err != nil {
			switch err := err.(type) {
			case transport.ConnectionError:
				// Nothing to do here.
			case transport.StreamError:
				if err := t.WriteStatus(stream, err.Code, err.Desc); err != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", err)
				}
			default:
				panic(fmt.Sprintf("grpc: Unexpected error (%T) from recvMsg: %v", err, err))
			}
			return err
		}

		if err := checkRecvPayload(pf, stream.RecvCompress(), s.opts.dc); err != nil {
			switch err := err.(type) {
			case transport.StreamError:
				if err := t.WriteStatus(stream, err.Code, err.Desc); err != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", err)
				}
			default:
				if err := t.WriteStatus(stream, codes.Internal, err.Error()); err != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", err)
				}

			}
			return err
		}
		statusCode := codes.OK
		statusDesc := ""
		df := func(v interface{}) error {
			if pf == compressionMade {
				var err error
				req, err = s.opts.dc.Do(bytes.NewReader(req))
				if err != nil {
					if err := t.WriteStatus(stream, codes.Internal, err.Error()); err != nil {
						grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", err)
					}
					return err
				}
			}
			if err := s.opts.codec.Unmarshal(req, v); err != nil {
				return err
			}
			if trInfo != nil {
				trInfo.tr.LazyLog(&payload{sent: false, msg: v}, true)
			}
			return nil
		}
		reply, appErr := md.Handler(srv.server, stream.Context(), df)
		if appErr != nil {
			if err, ok := appErr.(rpcError); ok {
				statusCode = err.code
				statusDesc = err.desc
			} else {
				statusCode = convertCode(appErr)
				statusDesc = appErr.Error()
			}
			if trInfo != nil && statusCode != codes.OK {
				trInfo.tr.LazyLog(stringer(statusDesc), true)
				trInfo.tr.SetError()
			}
			if err := t.WriteStatus(stream, statusCode, statusDesc); err != nil {
				grpclog.Printf("grpc: Server.processUnaryRPC failed to write status: %v", err)
				return err
			}
			return nil
		}
		if trInfo != nil {
			trInfo.tr.LazyLog(stringer("OK"), false)
		}
		opts := &transport.Options{
			Last:  true,
			Delay: false,
		}
		if s.opts.cp != nil {
			stream.SetSendCompress(s.opts.cp.Type())
		}
		if err := s.sendResponse(t, stream, reply, s.opts.cp, opts); err != nil {
			switch err := err.(type) {
			case transport.ConnectionError:
				// Nothing to do here.
			case transport.StreamError:
				statusCode = err.Code
				statusDesc = err.Desc
			default:
				statusCode = codes.Unknown
				statusDesc = err.Error()
			}
			return err
		}
		if trInfo != nil {
			trInfo.tr.LazyLog(&payload{sent: true, msg: reply}, true)
		}
		return t.WriteStatus(stream, statusCode, statusDesc)
	}
}

func (s *Server) processStreamingRPC(t transport.ServerTransport, stream *transport.Stream, srv *service, sd *StreamDesc, trInfo *traceInfo) (err error) {
	if s.opts.cp != nil {
		stream.SetSendCompress(s.opts.cp.Type())
	}
	ss := &serverStream{
		t:      t,
		s:      stream,
		p:      &parser{s: stream},
		codec:  s.opts.codec,
		cp:     s.opts.cp,
		dc:     s.opts.dc,
		trInfo: trInfo,
	}
	if ss.cp != nil {
		ss.cbuf = new(bytes.Buffer)
	}
	if trInfo != nil {
		trInfo.tr.LazyLog(&trInfo.firstLine, false)
		defer func() {
			ss.mu.Lock()
			if err != nil && err != io.EOF {
				ss.trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
				ss.trInfo.tr.SetError()
			}
			ss.trInfo.tr.Finish()
			ss.trInfo.tr = nil
			ss.mu.Unlock()
		}()
	}
	if appErr := sd.Handler(srv.server, ss); appErr != nil {
		if err, ok := appErr.(rpcError); ok {
			ss.statusCode = err.code
			ss.statusDesc = err.desc
		} else if err, ok := appErr.(transport.StreamError); ok {
			ss.statusCode = err.Code
			ss.statusDesc = err.Desc
		} else {
			ss.statusCode = convertCode(appErr)
			ss.statusDesc = appErr.Error()
		}
	}
	if trInfo != nil {
		ss.mu.Lock()
		if ss.statusCode != codes.OK {
			ss.trInfo.tr.LazyLog(stringer(ss.statusDesc), true)
			ss.trInfo.tr.SetError()
		} else {
			ss.trInfo.tr.LazyLog(stringer("OK"), false)
		}
		ss.mu.Unlock()
	}
	return t.WriteStatus(ss.s, ss.statusCode, ss.statusDesc)

}

func (s *Server) handleStream(t transport.ServerTransport, stream *transport.Stream, trInfo *traceInfo) {
	sm := stream.Method()
	if sm != "" && sm[0] == '/' {
		sm = sm[1:]
	}
	pos := strings.LastIndex(sm, "/")
	if pos == -1 {
		if trInfo != nil {
			trInfo.tr.LazyLog(&fmtStringer{"Malformed method name %q", []interface{}{sm}}, true)
			trInfo.tr.SetError()
		}
		if err := t.WriteStatus(stream, codes.InvalidArgument, fmt.Sprintf("malformed method name: %q", stream.Method())); err != nil {
			if trInfo != nil {
				trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
				trInfo.tr.SetError()
			}
			grpclog.Printf("grpc: Server.handleStream failed to write status: %v", err)
		}
		if trInfo != nil {
			trInfo.tr.Finish()
		}
		return
	}
	service := sm[:pos]
	method := sm[pos+1:]
	srv, ok := s.m[service]
	if !ok {
		if trInfo != nil {
			trInfo.tr.LazyLog(&fmtStringer{"Unknown service %v", []interface{}{service}}, true)
			trInfo.tr.SetError()
		}
		if err := t.WriteStatus(stream, codes.Unimplemented, fmt.Sprintf("unknown service %v", service)); err != nil {
			if trInfo != nil {
				trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
				trInfo.tr.SetError()
			}
			grpclog.Printf("grpc: Server.handleStream failed to write status: %v", err)
		}
		if trInfo != nil {
			trInfo.tr.Finish()
		}
		return
	}
	// Unary RPC or Streaming RPC?
	if md, ok := srv.md[method]; ok {
		s.processUnaryRPC(t, stream, srv, md, trInfo)
		return
	}
	if sd, ok := srv.sd[method]; ok {
		s.processStreamingRPC(t, stream, srv, sd, trInfo)
		return
	}
	if trInfo != nil {
		trInfo.tr.LazyLog(&fmtStringer{"Unknown method %v", []interface{}{method}}, true)
		trInfo.tr.SetError()
	}
	if err := t.WriteStatus(stream, codes.Unimplemented, fmt.Sprintf("unknown method %v", method)); err != nil {
		if trInfo != nil {
			trInfo.tr.LazyLog(&fmtStringer{"%v", []interface{}{err}}, true)
			trInfo.tr.SetError()
		}
		grpclog.Printf("grpc: Server.handleStream failed to write status: %v", err)
	}
	if trInfo != nil {
		trInfo.tr.Finish()
	}
}

// Stop stops the gRPC server. Once Stop returns, the server stops accepting
// connection requests and closes all the connected connections.
func (s *Server) Stop() {
	s.mu.Lock()
	listeners := s.lis
	s.lis = nil
	cs := s.conns
	s.conns = nil
	s.mu.Unlock()
	for lis := range listeners {
		lis.Close()
	}
	for c := range cs {
		c.Close()
	}
	s.mu.Lock()
	if s.events != nil {
		s.events.Finish()
		s.events = nil
	}
	s.mu.Unlock()
}

// TestingCloseConns closes all exiting transports but keeps s.lis accepting new
// connections. This is for test only now.
func (s *Server) TestingCloseConns() {
	s.mu.Lock()
	for c := range s.conns {
		c.Close()
	}
	s.conns = make(map[transport.ServerTransport]bool)
	s.mu.Unlock()
}

// SendHeader sends header metadata. It may be called at most once from a unary
// RPC handler. The ctx is the RPC handler's Context or one derived from it.
func SendHeader(ctx context.Context, md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	stream, ok := transport.StreamFromContext(ctx)
	if !ok {
		return fmt.Errorf("grpc: failed to fetch the stream from the context %v", ctx)
	}
	t := stream.ServerTransport()
	if t == nil {
		grpclog.Fatalf("grpc: SendHeader: %v has no ServerTransport to send header metadata.", stream)
	}
	return t.WriteHeader(stream, md)
}

// SetTrailer sets the trailer metadata that will be sent when an RPC returns.
// It may be called at most once from a unary RPC handler. The ctx is the RPC
// handler's Context or one derived from it.
func SetTrailer(ctx context.Context, md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	stream, ok := transport.StreamFromContext(ctx)
	if !ok {
		return fmt.Errorf("grpc: failed to fetch the stream from the context %v", ctx)
	}
	return stream.SetTrailer(md)
}
