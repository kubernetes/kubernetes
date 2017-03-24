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
	"net/http"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/tap"
	"google.golang.org/grpc/transport"
)

type methodHandler func(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor UnaryServerInterceptor) (interface{}, error)

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
	Metadata    interface{}
}

// service consists of the information of the server serving this service and
// the methods in this service.
type service struct {
	server interface{} // the server for service methods
	md     map[string]*MethodDesc
	sd     map[string]*StreamDesc
	mdata  interface{}
}

// Server is a gRPC server to serve RPC requests.
type Server struct {
	opts options

	mu     sync.Mutex // guards following
	lis    map[net.Listener]bool
	conns  map[io.Closer]bool
	drain  bool
	ctx    context.Context
	cancel context.CancelFunc
	// A CondVar to let GracefulStop() blocks until all the pending RPCs are finished
	// and all the transport goes away.
	cv     *sync.Cond
	m      map[string]*service // service name -> service info
	events trace.EventLog
}

type options struct {
	creds                credentials.TransportCredentials
	codec                Codec
	cp                   Compressor
	dc                   Decompressor
	maxMsgSize           int
	unaryInt             UnaryServerInterceptor
	streamInt            StreamServerInterceptor
	inTapHandle          tap.ServerInHandle
	maxConcurrentStreams uint32
	useHandlerImpl       bool // use http.Handler-based server
}

var defaultMaxMsgSize = 1024 * 1024 * 4 // use 4MB as the default message size limit

// A ServerOption sets options.
type ServerOption func(*options)

// CustomCodec returns a ServerOption that sets a codec for message marshaling and unmarshaling.
func CustomCodec(codec Codec) ServerOption {
	return func(o *options) {
		o.codec = codec
	}
}

// RPCCompressor returns a ServerOption that sets a compressor for outbound messages.
func RPCCompressor(cp Compressor) ServerOption {
	return func(o *options) {
		o.cp = cp
	}
}

// RPCDecompressor returns a ServerOption that sets a decompressor for inbound messages.
func RPCDecompressor(dc Decompressor) ServerOption {
	return func(o *options) {
		o.dc = dc
	}
}

// MaxMsgSize returns a ServerOption to set the max message size in bytes for inbound mesages.
// If this is not set, gRPC uses the default 4MB.
func MaxMsgSize(m int) ServerOption {
	return func(o *options) {
		o.maxMsgSize = m
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
func Creds(c credentials.TransportCredentials) ServerOption {
	return func(o *options) {
		o.creds = c
	}
}

// UnaryInterceptor returns a ServerOption that sets the UnaryServerInterceptor for the
// server. Only one unary interceptor can be installed. The construction of multiple
// interceptors (e.g., chaining) can be implemented at the caller.
func UnaryInterceptor(i UnaryServerInterceptor) ServerOption {
	return func(o *options) {
		if o.unaryInt != nil {
			panic("The unary server interceptor has been set.")
		}
		o.unaryInt = i
	}
}

// StreamInterceptor returns a ServerOption that sets the StreamServerInterceptor for the
// server. Only one stream interceptor can be installed.
func StreamInterceptor(i StreamServerInterceptor) ServerOption {
	return func(o *options) {
		if o.streamInt != nil {
			panic("The stream server interceptor has been set.")
		}
		o.streamInt = i
	}
}

// InTapHandle returns a ServerOption that sets the tap handle for all the server
// transport to be created. Only one can be installed.
func InTapHandle(h tap.ServerInHandle) ServerOption {
	return func(o *options) {
		if o.inTapHandle != nil {
			panic("The tap handle has been set.")
		}
		o.inTapHandle = h
	}
}

// NewServer creates a gRPC server which has no service registered and has not
// started to accept requests yet.
func NewServer(opt ...ServerOption) *Server {
	var opts options
	opts.maxMsgSize = defaultMaxMsgSize
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
		conns: make(map[io.Closer]bool),
		m:     make(map[string]*service),
	}
	s.cv = sync.NewCond(&s.mu)
	s.ctx, s.cancel = context.WithCancel(context.Background())
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
		mdata:  sd.Metadata,
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

// MethodInfo contains the information of an RPC including its method name and type.
type MethodInfo struct {
	// Name is the method name only, without the service name or package name.
	Name string
	// IsClientStream indicates whether the RPC is a client streaming RPC.
	IsClientStream bool
	// IsServerStream indicates whether the RPC is a server streaming RPC.
	IsServerStream bool
}

// ServiceInfo contains unary RPC method info, streaming RPC methid info and metadata for a service.
type ServiceInfo struct {
	Methods []MethodInfo
	// Metadata is the metadata specified in ServiceDesc when registering service.
	Metadata interface{}
}

// GetServiceInfo returns a map from service names to ServiceInfo.
// Service names include the package names, in the form of <package>.<service>.
func (s *Server) GetServiceInfo() map[string]ServiceInfo {
	ret := make(map[string]ServiceInfo)
	for n, srv := range s.m {
		methods := make([]MethodInfo, 0, len(srv.md)+len(srv.sd))
		for m := range srv.md {
			methods = append(methods, MethodInfo{
				Name:           m,
				IsClientStream: false,
				IsServerStream: false,
			})
		}
		for m, d := range srv.sd {
			methods = append(methods, MethodInfo{
				Name:           m,
				IsClientStream: d.ClientStreams,
				IsServerStream: d.ServerStreams,
			})
		}

		ret[n] = ServiceInfo{
			Methods:  methods,
			Metadata: srv.mdata,
		}
	}
	return ret
}

var (
	// ErrServerStopped indicates that the operation is now illegal because of
	// the server being stopped.
	ErrServerStopped = errors.New("grpc: the server has been stopped")
)

func (s *Server) useTransportAuthenticator(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	if s.opts.creds == nil {
		return rawConn, nil, nil
	}
	return s.opts.creds.ServerHandshake(rawConn)
}

// Serve accepts incoming connections on the listener lis, creating a new
// ServerTransport and service goroutine for each. The service goroutines
// read gRPC requests and then call the registered handlers to reply to them.
// Serve returns when lis.Accept fails with fatal errors.  lis will be closed when
// this method returns.
func (s *Server) Serve(lis net.Listener) error {
	s.mu.Lock()
	s.printf("serving")
	if s.lis == nil {
		s.mu.Unlock()
		lis.Close()
		return ErrServerStopped
	}
	s.lis[lis] = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		if s.lis != nil && s.lis[lis] {
			lis.Close()
			delete(s.lis, lis)
		}
		s.mu.Unlock()
	}()

	var tempDelay time.Duration // how long to sleep on accept failure

	for {
		rawConn, err := lis.Accept()
		if err != nil {
			if ne, ok := err.(interface {
				Temporary() bool
			}); ok && ne.Temporary() {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				s.mu.Lock()
				s.printf("Accept error: %v; retrying in %v", err, tempDelay)
				s.mu.Unlock()
				select {
				case <-time.After(tempDelay):
				case <-s.ctx.Done():
				}
				continue
			}
			s.mu.Lock()
			s.printf("done serving; Accept = %v", err)
			s.mu.Unlock()
			return err
		}
		tempDelay = 0
		// Start a new goroutine to deal with rawConn
		// so we don't stall this Accept loop goroutine.
		go s.handleRawConn(rawConn)
	}
}

// handleRawConn is run in its own goroutine and handles a just-accepted
// connection that has not had any I/O performed on it yet.
func (s *Server) handleRawConn(rawConn net.Conn) {
	conn, authInfo, err := s.useTransportAuthenticator(rawConn)
	if err != nil {
		s.mu.Lock()
		s.errorf("ServerHandshake(%q) failed: %v", rawConn.RemoteAddr(), err)
		s.mu.Unlock()
		grpclog.Printf("grpc: Server.Serve failed to complete security handshake from %q: %v", rawConn.RemoteAddr(), err)
		// If serverHandShake returns ErrConnDispatched, keep rawConn open.
		if err != credentials.ErrConnDispatched {
			rawConn.Close()
		}
		return
	}

	s.mu.Lock()
	if s.conns == nil {
		s.mu.Unlock()
		conn.Close()
		return
	}
	s.mu.Unlock()

	if s.opts.useHandlerImpl {
		s.serveUsingHandler(conn)
	} else {
		s.serveHTTP2Transport(conn, authInfo)
	}
}

// serveHTTP2Transport sets up a http/2 transport (using the
// gRPC http2 server transport in transport/http2_server.go) and
// serves streams on it.
// This is run in its own goroutine (it does network I/O in
// transport.NewServerTransport).
func (s *Server) serveHTTP2Transport(c net.Conn, authInfo credentials.AuthInfo) {
	config := &transport.ServerConfig{
		MaxStreams:  s.opts.maxConcurrentStreams,
		AuthInfo:    authInfo,
		InTapHandle: s.opts.inTapHandle,
	}
	st, err := transport.NewServerTransport("http2", c, config)
	if err != nil {
		s.mu.Lock()
		s.errorf("NewServerTransport(%q) failed: %v", c.RemoteAddr(), err)
		s.mu.Unlock()
		c.Close()
		grpclog.Println("grpc: Server.Serve failed to create ServerTransport: ", err)
		return
	}
	if !s.addConn(st) {
		st.Close()
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
	}, func(ctx context.Context, method string) context.Context {
		if !EnableTracing {
			return ctx
		}
		tr := trace.New("grpc.Recv."+methodFamily(method), method)
		return trace.NewContext(ctx, tr)
	})
	wg.Wait()
}

var _ http.Handler = (*Server)(nil)

// serveUsingHandler is called from handleRawConn when s is configured
// to handle requests via the http.Handler interface. It sets up a
// net/http.Server to handle the just-accepted conn. The http.Server
// is configured to route all incoming requests (all HTTP/2 streams)
// to ServeHTTP, which creates a new ServerTransport for each stream.
// serveUsingHandler blocks until conn closes.
//
// This codepath is only used when Server.TestingUseHandlerImpl has
// been configured. This lets the end2end tests exercise the ServeHTTP
// method as one of the environment types.
//
// conn is the *tls.Conn that's already been authenticated.
func (s *Server) serveUsingHandler(conn net.Conn) {
	if !s.addConn(conn) {
		conn.Close()
		return
	}
	defer s.removeConn(conn)
	h2s := &http2.Server{
		MaxConcurrentStreams: s.opts.maxConcurrentStreams,
	}
	h2s.ServeConn(conn, &http2.ServeConnOpts{
		Handler: s,
	})
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	st, err := transport.NewServerHandlerTransport(w, r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !s.addConn(st) {
		st.Close()
		return
	}
	defer s.removeConn(st)
	s.serveStreams(st)
}

// traceInfo returns a traceInfo and associates it with stream, if tracing is enabled.
// If tracing is not enabled, it returns nil.
func (s *Server) traceInfo(st transport.ServerTransport, stream *transport.Stream) (trInfo *traceInfo) {
	tr, ok := trace.FromContext(stream.Context())
	if !ok {
		return nil
	}

	trInfo = &traceInfo{
		tr: tr,
	}
	trInfo.firstLine.client = false
	trInfo.firstLine.remoteAddr = st.RemoteAddr()

	if dl, ok := stream.Context().Deadline(); ok {
		trInfo.firstLine.deadline = dl.Sub(time.Now())
	}
	return trInfo
}

func (s *Server) addConn(c io.Closer) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conns == nil || s.drain {
		return false
	}
	s.conns[c] = true
	return true
}

func (s *Server) removeConn(c io.Closer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conns != nil {
		delete(s.conns, c)
		s.cv.Broadcast()
	}
}

func (s *Server) sendResponse(t transport.ServerTransport, stream *transport.Stream, msg interface{}, cp Compressor, opts *transport.Options) error {
	var (
		cbuf       *bytes.Buffer
		outPayload *stats.OutPayload
	)
	if cp != nil {
		cbuf = new(bytes.Buffer)
	}
	if stats.On() {
		outPayload = &stats.OutPayload{}
	}
	p, err := encode(s.opts.codec, msg, cp, cbuf, outPayload)
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
	err = t.Write(stream, p, opts)
	if err == nil && outPayload != nil {
		outPayload.SentTime = time.Now()
		stats.HandleRPC(stream.Context(), outPayload)
	}
	return err
}

func (s *Server) processUnaryRPC(t transport.ServerTransport, stream *transport.Stream, srv *service, md *MethodDesc, trInfo *traceInfo) (err error) {
	if stats.On() {
		begin := &stats.Begin{
			BeginTime: time.Now(),
		}
		stats.HandleRPC(stream.Context(), begin)
	}
	defer func() {
		if stats.On() {
			end := &stats.End{
				EndTime: time.Now(),
			}
			if err != nil && err != io.EOF {
				end.Error = toRPCErr(err)
			}
			stats.HandleRPC(stream.Context(), end)
		}
	}()
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
	if s.opts.cp != nil {
		// NOTE: this needs to be ahead of all handling, https://github.com/grpc/grpc-go/issues/686.
		stream.SetSendCompress(s.opts.cp.Type())
	}
	p := &parser{r: stream}
	for {
		pf, req, err := p.recvMsg(s.opts.maxMsgSize)
		if err == io.EOF {
			// The entire stream is done (for unary RPC only).
			return err
		}
		if err == io.ErrUnexpectedEOF {
			err = Errorf(codes.Internal, io.ErrUnexpectedEOF.Error())
		}
		if err != nil {
			switch err := err.(type) {
			case *rpcError:
				if e := t.WriteStatus(stream, err.code, err.desc); e != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", e)
				}
			case transport.ConnectionError:
				// Nothing to do here.
			case transport.StreamError:
				if e := t.WriteStatus(stream, err.Code, err.Desc); e != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", e)
				}
			default:
				panic(fmt.Sprintf("grpc: Unexpected error (%T) from recvMsg: %v", err, err))
			}
			return err
		}

		if err := checkRecvPayload(pf, stream.RecvCompress(), s.opts.dc); err != nil {
			switch err := err.(type) {
			case *rpcError:
				if e := t.WriteStatus(stream, err.code, err.desc); e != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", e)
				}
				return err
			default:
				if e := t.WriteStatus(stream, codes.Internal, err.Error()); e != nil {
					grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", e)
				}
				// TODO checkRecvPayload always return RPC error. Add a return here if necessary.
			}
		}
		var inPayload *stats.InPayload
		if stats.On() {
			inPayload = &stats.InPayload{
				RecvTime: time.Now(),
			}
		}
		statusCode := codes.OK
		statusDesc := ""
		df := func(v interface{}) error {
			if inPayload != nil {
				inPayload.WireLength = len(req)
			}
			if pf == compressionMade {
				var err error
				req, err = s.opts.dc.Do(bytes.NewReader(req))
				if err != nil {
					if err := t.WriteStatus(stream, codes.Internal, err.Error()); err != nil {
						grpclog.Printf("grpc: Server.processUnaryRPC failed to write status %v", err)
					}
					return Errorf(codes.Internal, err.Error())
				}
			}
			if len(req) > s.opts.maxMsgSize {
				// TODO: Revisit the error code. Currently keep it consistent with
				// java implementation.
				statusCode = codes.Internal
				statusDesc = fmt.Sprintf("grpc: server received a message of %d bytes exceeding %d limit", len(req), s.opts.maxMsgSize)
			}
			if err := s.opts.codec.Unmarshal(req, v); err != nil {
				return err
			}
			if inPayload != nil {
				inPayload.Payload = v
				inPayload.Data = req
				inPayload.Length = len(req)
				stats.HandleRPC(stream.Context(), inPayload)
			}
			if trInfo != nil {
				trInfo.tr.LazyLog(&payload{sent: false, msg: v}, true)
			}
			return nil
		}
		reply, appErr := md.Handler(srv.server, stream.Context(), df, s.opts.unaryInt)
		if appErr != nil {
			if err, ok := appErr.(*rpcError); ok {
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
			}
			return Errorf(statusCode, statusDesc)
		}
		if trInfo != nil {
			trInfo.tr.LazyLog(stringer("OK"), false)
		}
		opts := &transport.Options{
			Last:  true,
			Delay: false,
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
		errWrite := t.WriteStatus(stream, statusCode, statusDesc)
		if statusCode != codes.OK {
			return Errorf(statusCode, statusDesc)
		}
		return errWrite
	}
}

func (s *Server) processStreamingRPC(t transport.ServerTransport, stream *transport.Stream, srv *service, sd *StreamDesc, trInfo *traceInfo) (err error) {
	if stats.On() {
		begin := &stats.Begin{
			BeginTime: time.Now(),
		}
		stats.HandleRPC(stream.Context(), begin)
	}
	defer func() {
		if stats.On() {
			end := &stats.End{
				EndTime: time.Now(),
			}
			if err != nil && err != io.EOF {
				end.Error = toRPCErr(err)
			}
			stats.HandleRPC(stream.Context(), end)
		}
	}()
	if s.opts.cp != nil {
		stream.SetSendCompress(s.opts.cp.Type())
	}
	ss := &serverStream{
		t:          t,
		s:          stream,
		p:          &parser{r: stream},
		codec:      s.opts.codec,
		cp:         s.opts.cp,
		dc:         s.opts.dc,
		maxMsgSize: s.opts.maxMsgSize,
		trInfo:     trInfo,
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
	var appErr error
	if s.opts.streamInt == nil {
		appErr = sd.Handler(srv.server, ss)
	} else {
		info := &StreamServerInfo{
			FullMethod:     stream.Method(),
			IsClientStream: sd.ClientStreams,
			IsServerStream: sd.ServerStreams,
		}
		appErr = s.opts.streamInt(srv.server, ss, info, sd.Handler)
	}
	if appErr != nil {
		if err, ok := appErr.(*rpcError); ok {
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
	errWrite := t.WriteStatus(ss.s, ss.statusCode, ss.statusDesc)
	if ss.statusCode != codes.OK {
		return Errorf(ss.statusCode, ss.statusDesc)
	}
	return errWrite

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
		errDesc := fmt.Sprintf("malformed method name: %q", stream.Method())
		if err := t.WriteStatus(stream, codes.InvalidArgument, errDesc); err != nil {
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
		errDesc := fmt.Sprintf("unknown service %v", service)
		if err := t.WriteStatus(stream, codes.Unimplemented, errDesc); err != nil {
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
	errDesc := fmt.Sprintf("unknown method %v", method)
	if err := t.WriteStatus(stream, codes.Unimplemented, errDesc); err != nil {
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

// Stop stops the gRPC server. It immediately closes all open
// connections and listeners.
// It cancels all active RPCs on the server side and the corresponding
// pending RPCs on the client side will get notified by connection
// errors.
func (s *Server) Stop() {
	s.mu.Lock()
	listeners := s.lis
	s.lis = nil
	st := s.conns
	s.conns = nil
	// interrupt GracefulStop if Stop and GracefulStop are called concurrently.
	s.cv.Broadcast()
	s.mu.Unlock()

	for lis := range listeners {
		lis.Close()
	}
	for c := range st {
		c.Close()
	}

	s.mu.Lock()
	s.cancel()
	if s.events != nil {
		s.events.Finish()
		s.events = nil
	}
	s.mu.Unlock()
}

// GracefulStop stops the gRPC server gracefully. It stops the server to accept new
// connections and RPCs and blocks until all the pending RPCs are finished.
func (s *Server) GracefulStop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conns == nil {
		return
	}
	for lis := range s.lis {
		lis.Close()
	}
	s.lis = nil
	s.cancel()
	if !s.drain {
		for c := range s.conns {
			c.(transport.ServerTransport).Drain()
		}
		s.drain = true
	}
	for len(s.conns) != 0 {
		s.cv.Wait()
	}
	s.conns = nil
	if s.events != nil {
		s.events.Finish()
		s.events = nil
	}
}

func init() {
	internal.TestingCloseConns = func(arg interface{}) {
		arg.(*Server).testingCloseConns()
	}
	internal.TestingUseHandlerImpl = func(arg interface{}) {
		arg.(*Server).opts.useHandlerImpl = true
	}
}

// testingCloseConns closes all existing transports but keeps s.lis
// accepting new connections.
func (s *Server) testingCloseConns() {
	s.mu.Lock()
	for c := range s.conns {
		c.Close()
		delete(s.conns, c)
	}
	s.mu.Unlock()
}

// SetHeader sets the header metadata.
// When called multiple times, all the provided metadata will be merged.
// All the metadata will be sent out when one of the following happens:
//  - grpc.SendHeader() is called;
//  - The first response is sent out;
//  - An RPC status is sent out (error or success).
func SetHeader(ctx context.Context, md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	stream, ok := transport.StreamFromContext(ctx)
	if !ok {
		return Errorf(codes.Internal, "grpc: failed to fetch the stream from the context %v", ctx)
	}
	return stream.SetHeader(md)
}

// SendHeader sends header metadata. It may be called at most once.
// The provided md and headers set by SetHeader() will be sent.
func SendHeader(ctx context.Context, md metadata.MD) error {
	stream, ok := transport.StreamFromContext(ctx)
	if !ok {
		return Errorf(codes.Internal, "grpc: failed to fetch the stream from the context %v", ctx)
	}
	t := stream.ServerTransport()
	if t == nil {
		grpclog.Fatalf("grpc: SendHeader: %v has no ServerTransport to send header metadata.", stream)
	}
	if err := t.WriteHeader(stream, md); err != nil {
		return toRPCErr(err)
	}
	return nil
}

// SetTrailer sets the trailer metadata that will be sent when an RPC returns.
// When called more than once, all the provided metadata will be merged.
func SetTrailer(ctx context.Context, md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	stream, ok := transport.StreamFromContext(ctx)
	if !ok {
		return Errorf(codes.Internal, "grpc: failed to fetch the stream from the context %v", ctx)
	}
	return stream.SetTrailer(md)
}
