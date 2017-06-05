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

package etcdmain

import (
	"crypto/tls"
	"io/ioutil"
	defaultLog "log"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/transport"

	"github.com/cockroachdb/cmux"
	gw "github.com/grpc-ecosystem/grpc-gateway/runtime"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type serveCtx struct {
	l        net.Listener
	host     string
	secure   bool
	insecure bool
}

// serve accepts incoming connections on the listener l,
// creating a new service goroutine for each. The service goroutines
// read requests and then call handler to reply to them.
func serve(sctx *serveCtx, s *etcdserver.EtcdServer, tlscfg *tls.Config, handler http.Handler) error {
	logger := defaultLog.New(ioutil.Discard, "etcdhttp", 0)

	<-s.ReadyNotify()
	plog.Info("ready to serve client requests")

	m := cmux.New(sctx.l)

	if sctx.insecure {
		gs := v3rpc.Server(s, nil)
		grpcl := m.Match(cmux.HTTP2())
		go func() { plog.Fatal(gs.Serve(grpcl)) }()

		opts := []grpc.DialOption{
			grpc.WithInsecure(),
		}
		gwmux, err := registerGateway(sctx.l.Addr().String(), opts)
		if err != nil {
			return err
		}

		httpmux := http.NewServeMux()
		httpmux.Handle("/v3alpha/", gwmux)
		httpmux.Handle("/", handler)
		srvhttp := &http.Server{
			Handler:  httpmux,
			ErrorLog: logger, // do not log user error
		}
		httpl := m.Match(cmux.HTTP1())
		go func() { plog.Fatal(srvhttp.Serve(httpl)) }()
		plog.Noticef("serving insecure client requests on %s, this is strongly discouraged!", sctx.host)
	}

	if sctx.secure {
		gs := v3rpc.Server(s, tlscfg)
		handler = grpcHandlerFunc(gs, handler)

		dtls := transport.ShallowCopyTLSConfig(tlscfg)
		// trust local server
		dtls.InsecureSkipVerify = true
		creds := credentials.NewTLS(dtls)
		opts := []grpc.DialOption{grpc.WithTransportCredentials(creds)}
		gwmux, err := registerGateway(sctx.l.Addr().String(), opts)
		if err != nil {
			return err
		}

		tlsl := tls.NewListener(m.Match(cmux.Any()), tlscfg)
		// TODO: add debug flag; enable logging when debug flag is set
		httpmux := http.NewServeMux()
		httpmux.Handle("/v3alpha/", gwmux)
		httpmux.Handle("/", handler)
		srv := &http.Server{
			Handler:   httpmux,
			TLSConfig: tlscfg,
			ErrorLog:  logger, // do not log user error
		}
		go func() { plog.Fatal(srv.Serve(tlsl)) }()

		plog.Infof("serving client requests on %s", sctx.host)
	}

	return m.Serve()
}

// grpcHandlerFunc returns an http.Handler that delegates to grpcServer on incoming gRPC
// connections or otherHandler otherwise. Copied from cockroachdb.
func grpcHandlerFunc(grpcServer *grpc.Server, otherHandler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.Contains(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
		} else {
			otherHandler.ServeHTTP(w, r)
		}
	})
}

func servePeerHTTP(l net.Listener, handler http.Handler) error {
	logger := defaultLog.New(ioutil.Discard, "etcdhttp", 0)
	// TODO: add debug flag; enable logging when debug flag is set
	srv := &http.Server{
		Handler:     handler,
		ReadTimeout: 5 * time.Minute,
		ErrorLog:    logger, // do not log user error
	}
	return srv.Serve(l)
}

func registerGateway(addr string, opts []grpc.DialOption) (*gw.ServeMux, error) {
	gwmux := gw.NewServeMux()

	err := pb.RegisterKVHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	err = pb.RegisterWatchHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	err = pb.RegisterLeaseHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	err = pb.RegisterClusterHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	err = pb.RegisterMaintenanceHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	err = pb.RegisterAuthHandlerFromEndpoint(context.Background(), gwmux, addr, opts)
	if err != nil {
		return nil, err
	}
	return gwmux, nil
}
