// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package embed

import (
	"crypto/tls"
	"io/ioutil"
	defaultLog "log"
	"net"
	"net/http"
	"strings"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3client"
	"github.com/coreos/etcd/etcdserver/api/v3election"
	"github.com/coreos/etcd/etcdserver/api/v3election/v3electionpb"
	v3electiongw "github.com/coreos/etcd/etcdserver/api/v3election/v3electionpb/gw"
	"github.com/coreos/etcd/etcdserver/api/v3lock"
	"github.com/coreos/etcd/etcdserver/api/v3lock/v3lockpb"
	v3lockgw "github.com/coreos/etcd/etcdserver/api/v3lock/v3lockpb/gw"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	etcdservergw "github.com/coreos/etcd/etcdserver/etcdserverpb/gw"
	"github.com/coreos/etcd/pkg/debugutil"

	"github.com/cockroachdb/cmux"
	gw "github.com/grpc-ecosystem/grpc-gateway/runtime"
	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type serveCtx struct {
	l        net.Listener
	addr     string
	secure   bool
	insecure bool

	ctx    context.Context
	cancel context.CancelFunc

	userHandlers    map[string]http.Handler
	serviceRegister func(*grpc.Server)
	grpcServerC     chan *grpc.Server
}

func newServeCtx() *serveCtx {
	ctx, cancel := context.WithCancel(context.Background())
	return &serveCtx{ctx: ctx, cancel: cancel, userHandlers: make(map[string]http.Handler),
		grpcServerC: make(chan *grpc.Server, 2), // in case sctx.insecure,sctx.secure true
	}
}

// serve accepts incoming connections on the listener l,
// creating a new service goroutine for each. The service goroutines
// read requests and then call handler to reply to them.
func (sctx *serveCtx) serve(
	s *etcdserver.EtcdServer,
	tlscfg *tls.Config,
	handler http.Handler,
	errHandler func(error),
	gopts ...grpc.ServerOption) error {
	logger := defaultLog.New(ioutil.Discard, "etcdhttp", 0)
	<-s.ReadyNotify()
	plog.Info("ready to serve client requests")

	m := cmux.New(sctx.l)
	v3c := v3client.New(s)
	servElection := v3election.NewElectionServer(v3c)
	servLock := v3lock.NewLockServer(v3c)

	if sctx.insecure {
		gs := v3rpc.Server(s, nil, gopts...)
		sctx.grpcServerC <- gs
		v3electionpb.RegisterElectionServer(gs, servElection)
		v3lockpb.RegisterLockServer(gs, servLock)
		if sctx.serviceRegister != nil {
			sctx.serviceRegister(gs)
		}
		grpcl := m.Match(cmux.HTTP2())
		go func() { errHandler(gs.Serve(grpcl)) }()

		opts := []grpc.DialOption{
			grpc.WithInsecure(),
		}
		gwmux, err := sctx.registerGateway(opts)
		if err != nil {
			return err
		}

		httpmux := sctx.createMux(gwmux, handler)

		srvhttp := &http.Server{
			Handler:  httpmux,
			ErrorLog: logger, // do not log user error
		}
		httpl := m.Match(cmux.HTTP1())
		go func() { errHandler(srvhttp.Serve(httpl)) }()
		plog.Noticef("serving insecure client requests on %s, this is strongly discouraged!", sctx.l.Addr().String())
	}

	if sctx.secure {
		gs := v3rpc.Server(s, tlscfg, gopts...)
		sctx.grpcServerC <- gs
		v3electionpb.RegisterElectionServer(gs, servElection)
		v3lockpb.RegisterLockServer(gs, servLock)
		if sctx.serviceRegister != nil {
			sctx.serviceRegister(gs)
		}
		handler = grpcHandlerFunc(gs, handler)

		dtls := tlscfg.Clone()
		// trust local server
		dtls.InsecureSkipVerify = true
		creds := credentials.NewTLS(dtls)
		opts := []grpc.DialOption{grpc.WithTransportCredentials(creds)}
		gwmux, err := sctx.registerGateway(opts)
		if err != nil {
			return err
		}

		tlsl := tls.NewListener(m.Match(cmux.Any()), tlscfg)
		// TODO: add debug flag; enable logging when debug flag is set
		httpmux := sctx.createMux(gwmux, handler)

		srv := &http.Server{
			Handler:   httpmux,
			TLSConfig: tlscfg,
			ErrorLog:  logger, // do not log user error
		}
		go func() { errHandler(srv.Serve(tlsl)) }()

		plog.Infof("serving client requests on %s", sctx.l.Addr().String())
	}

	close(sctx.grpcServerC)
	return m.Serve()
}

// grpcHandlerFunc returns an http.Handler that delegates to grpcServer on incoming gRPC
// connections or otherHandler otherwise. Copied from cockroachdb.
func grpcHandlerFunc(grpcServer *grpc.Server, otherHandler http.Handler) http.Handler {
	if otherHandler == nil {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			grpcServer.ServeHTTP(w, r)
		})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.Contains(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
		} else {
			otherHandler.ServeHTTP(w, r)
		}
	})
}

type registerHandlerFunc func(context.Context, *gw.ServeMux, *grpc.ClientConn) error

func (sctx *serveCtx) registerGateway(opts []grpc.DialOption) (*gw.ServeMux, error) {
	ctx := sctx.ctx
	conn, err := grpc.DialContext(ctx, sctx.addr, opts...)
	if err != nil {
		return nil, err
	}
	gwmux := gw.NewServeMux()

	handlers := []registerHandlerFunc{
		etcdservergw.RegisterKVHandler,
		etcdservergw.RegisterWatchHandler,
		etcdservergw.RegisterLeaseHandler,
		etcdservergw.RegisterClusterHandler,
		etcdservergw.RegisterMaintenanceHandler,
		etcdservergw.RegisterAuthHandler,
		v3lockgw.RegisterLockHandler,
		v3electiongw.RegisterElectionHandler,
	}
	for _, h := range handlers {
		if err := h(ctx, gwmux, conn); err != nil {
			return nil, err
		}
	}
	go func() {
		<-ctx.Done()
		if cerr := conn.Close(); cerr != nil {
			plog.Warningf("failed to close conn to %s: %v", sctx.l.Addr().String(), cerr)
		}
	}()

	return gwmux, nil
}

func (sctx *serveCtx) createMux(gwmux *gw.ServeMux, handler http.Handler) *http.ServeMux {
	httpmux := http.NewServeMux()
	for path, h := range sctx.userHandlers {
		httpmux.Handle(path, h)
	}

	httpmux.Handle("/v3alpha/", gwmux)
	if handler != nil {
		httpmux.Handle("/", handler)
	}
	return httpmux
}

func (sctx *serveCtx) registerUserHandler(s string, h http.Handler) {
	if sctx.userHandlers[s] != nil {
		plog.Warningf("path %s already registered by user handler", s)
		return
	}
	sctx.userHandlers[s] = h
}

func (sctx *serveCtx) registerPprof() {
	for p, h := range debugutil.PProfHandlers() {
		sctx.registerUserHandler(p, h)
	}
}

func (sctx *serveCtx) registerTrace() {
	reqf := func(w http.ResponseWriter, r *http.Request) { trace.Render(w, r, true) }
	sctx.registerUserHandler("/debug/requests", http.HandlerFunc(reqf))
	evf := func(w http.ResponseWriter, r *http.Request) { trace.RenderEvents(w, r, true) }
	sctx.registerUserHandler("/debug/events", http.HandlerFunc(evf))
}
