// Copyright 2016 The etcd Authors
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
	"context"
	"crypto/tls"
	"fmt"
	"io/ioutil"
	defaultLog "log"
	"net"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/etcdhttp"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/debugutil"
	runtimeutil "github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/rafthttp"
	"github.com/coreos/pkg/capnslog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
)

var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "embed")

const (
	// internal fd usage includes disk usage and transport usage.
	// To read/write snapshot, snap pkg needs 1. In normal case, wal pkg needs
	// at most 2 to read/lock/write WALs. One case that it needs to 2 is to
	// read all logs after some snapshot index, which locates at the end of
	// the second last and the head of the last. For purging, it needs to read
	// directory, so it needs 1. For fd monitor, it needs 1.
	// For transport, rafthttp builds two long-polling connections and at most
	// four temporary connections with each member. There are at most 9 members
	// in a cluster, so it should reserve 96.
	// For the safety, we set the total reserved number to 150.
	reservedInternalFDNum = 150
)

// Etcd contains a running etcd server and its listeners.
type Etcd struct {
	Peers   []*peerListener
	Clients []net.Listener
	Server  *etcdserver.EtcdServer

	cfg   Config
	stopc chan struct{}
	errc  chan error
	sctxs map[string]*serveCtx

	closeOnce sync.Once
}

type peerListener struct {
	net.Listener
	serve func() error
	close func(context.Context) error
}

// StartEtcd launches the etcd server and HTTP handlers for client/server communication.
// The returned Etcd.Server is not guaranteed to have joined the cluster. Wait
// on the Etcd.Server.ReadyNotify() channel to know when it completes and is ready for use.
func StartEtcd(inCfg *Config) (e *Etcd, err error) {
	if err = inCfg.Validate(); err != nil {
		return nil, err
	}
	serving := false
	e = &Etcd{cfg: *inCfg, stopc: make(chan struct{})}
	cfg := &e.cfg
	defer func() {
		if e == nil || err == nil {
			return
		}
		if !serving {
			// errored before starting gRPC server for serveCtx.grpcServerC
			for _, sctx := range e.sctxs {
				close(sctx.grpcServerC)
			}
		}
		e.Close()
		e = nil
	}()

	if e.Peers, err = startPeerListeners(cfg); err != nil {
		return
	}
	if e.sctxs, err = startClientListeners(cfg); err != nil {
		return
	}
	for _, sctx := range e.sctxs {
		e.Clients = append(e.Clients, sctx.l)
	}

	var (
		urlsmap types.URLsMap
		token   string
	)

	if !isMemberInitialized(cfg) {
		urlsmap, token, err = cfg.PeerURLsMapAndToken("etcd")
		if err != nil {
			return e, fmt.Errorf("error setting up initial cluster: %v", err)
		}
	}

	srvcfg := &etcdserver.ServerConfig{
		Name:                    cfg.Name,
		ClientURLs:              cfg.ACUrls,
		PeerURLs:                cfg.APUrls,
		DataDir:                 cfg.Dir,
		DedicatedWALDir:         cfg.WalDir,
		SnapCount:               cfg.SnapCount,
		MaxSnapFiles:            cfg.MaxSnapFiles,
		MaxWALFiles:             cfg.MaxWalFiles,
		InitialPeerURLsMap:      urlsmap,
		InitialClusterToken:     token,
		DiscoveryURL:            cfg.Durl,
		DiscoveryProxy:          cfg.Dproxy,
		NewCluster:              cfg.IsNewCluster(),
		ForceNewCluster:         cfg.ForceNewCluster,
		PeerTLSInfo:             cfg.PeerTLSInfo,
		TickMs:                  cfg.TickMs,
		ElectionTicks:           cfg.ElectionTicks(),
		AutoCompactionRetention: cfg.AutoCompactionRetention,
		QuotaBackendBytes:       cfg.QuotaBackendBytes,
		MaxRequestBytes:         cfg.MaxRequestBytes,
		StrictReconfigCheck:     cfg.StrictReconfigCheck,
		ClientCertAuthEnabled:   cfg.ClientTLSInfo.ClientCertAuth,
		AuthToken:               cfg.AuthToken,
	}

	if e.Server, err = etcdserver.NewServer(srvcfg); err != nil {
		return
	}

	// configure peer handlers after rafthttp.Transport started
	ph := etcdhttp.NewPeerHandler(e.Server)
	for _, p := range e.Peers {
		srv := &http.Server{
			Handler:     ph,
			ReadTimeout: 5 * time.Minute,
			ErrorLog:    defaultLog.New(ioutil.Discard, "", 0), // do not log user error
		}

		l := p.Listener
		p.serve = func() error { return srv.Serve(l) }
		p.close = func(ctx context.Context) error {
			// gracefully shutdown http.Server
			// close open listeners, idle connections
			// until context cancel or time-out
			return srv.Shutdown(ctx)
		}
	}

	// buffer channel so goroutines on closed connections won't wait forever
	e.errc = make(chan error, len(e.Peers)+len(e.Clients)+2*len(e.sctxs))

	e.Server.Start()
	if err = e.serve(); err != nil {
		return
	}
	serving = true
	return
}

// Config returns the current configuration.
func (e *Etcd) Config() Config {
	return e.cfg
}

func (e *Etcd) Close() {
	e.closeOnce.Do(func() { close(e.stopc) })

	timeout := 2 * time.Second
	if e.Server != nil {
		timeout = e.Server.Cfg.ReqTimeout()
	}
	for _, sctx := range e.sctxs {
		for gs := range sctx.grpcServerC {
			ch := make(chan struct{})
			go func() {
				defer close(ch)
				// close listeners to stop accepting new connections,
				// will block on any existing transports
				gs.GracefulStop()
			}()
			// wait until all pending RPCs are finished
			select {
			case <-ch:
			case <-time.After(timeout):
				// took too long, manually close open transports
				// e.g. watch streams
				gs.Stop()
				// concurrent GracefulStop should be interrupted
				<-ch
			}
		}
	}

	for _, sctx := range e.sctxs {
		sctx.cancel()
	}
	for i := range e.Clients {
		if e.Clients[i] != nil {
			e.Clients[i].Close()
		}
	}

	// close rafthttp transports
	if e.Server != nil {
		e.Server.Stop()
	}

	// close all idle connections in peer handler (wait up to 1-second)
	for i := range e.Peers {
		if e.Peers[i] != nil && e.Peers[i].close != nil {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			e.Peers[i].close(ctx)
			cancel()
		}
	}
}

func (e *Etcd) Err() <-chan error { return e.errc }

func startPeerListeners(cfg *Config) (peers []*peerListener, err error) {
	if cfg.PeerAutoTLS && cfg.PeerTLSInfo.Empty() {
		phosts := make([]string, len(cfg.LPUrls))
		for i, u := range cfg.LPUrls {
			phosts[i] = u.Host
		}
		cfg.PeerTLSInfo, err = transport.SelfCert(filepath.Join(cfg.Dir, "fixtures", "peer"), phosts)
		if err != nil {
			plog.Fatalf("could not get certs (%v)", err)
		}
	} else if cfg.PeerAutoTLS {
		plog.Warningf("ignoring peer auto TLS since certs given")
	}

	if !cfg.PeerTLSInfo.Empty() {
		plog.Infof("peerTLS: %s", cfg.PeerTLSInfo)
	}

	peers = make([]*peerListener, len(cfg.LPUrls))
	defer func() {
		if err == nil {
			return
		}
		for i := range peers {
			if peers[i] != nil && peers[i].close != nil {
				plog.Info("stopping listening for peers on ", cfg.LPUrls[i].String())
				peers[i].close(context.Background())
			}
		}
	}()

	for i, u := range cfg.LPUrls {
		if u.Scheme == "http" {
			if !cfg.PeerTLSInfo.Empty() {
				plog.Warningf("The scheme of peer url %s is HTTP while peer key/cert files are presented. Ignored peer key/cert files.", u.String())
			}
			if cfg.PeerTLSInfo.ClientCertAuth {
				plog.Warningf("The scheme of peer url %s is HTTP while client cert auth (--peer-client-cert-auth) is enabled. Ignored client cert auth for this url.", u.String())
			}
		}
		peers[i] = &peerListener{close: func(context.Context) error { return nil }}
		peers[i].Listener, err = rafthttp.NewListener(u, &cfg.PeerTLSInfo)
		if err != nil {
			return nil, err
		}
		// once serve, overwrite with 'http.Server.Shutdown'
		peers[i].close = func(context.Context) error {
			return peers[i].Listener.Close()
		}
		plog.Info("listening for peers on ", u.String())
	}
	return peers, nil
}

func startClientListeners(cfg *Config) (sctxs map[string]*serveCtx, err error) {
	if cfg.ClientAutoTLS && cfg.ClientTLSInfo.Empty() {
		chosts := make([]string, len(cfg.LCUrls))
		for i, u := range cfg.LCUrls {
			chosts[i] = u.Host
		}
		cfg.ClientTLSInfo, err = transport.SelfCert(filepath.Join(cfg.Dir, "fixtures", "client"), chosts)
		if err != nil {
			plog.Fatalf("could not get certs (%v)", err)
		}
	} else if cfg.ClientAutoTLS {
		plog.Warningf("ignoring client auto TLS since certs given")
	}

	if cfg.EnablePprof {
		plog.Infof("pprof is enabled under %s", debugutil.HTTPPrefixPProf)
	}

	sctxs = make(map[string]*serveCtx)
	for _, u := range cfg.LCUrls {
		sctx := newServeCtx()

		if u.Scheme == "http" || u.Scheme == "unix" {
			if !cfg.ClientTLSInfo.Empty() {
				plog.Warningf("The scheme of client url %s is HTTP while peer key/cert files are presented. Ignored key/cert files.", u.String())
			}
			if cfg.ClientTLSInfo.ClientCertAuth {
				plog.Warningf("The scheme of client url %s is HTTP while client cert auth (--client-cert-auth) is enabled. Ignored client cert auth for this url.", u.String())
			}
		}
		if (u.Scheme == "https" || u.Scheme == "unixs") && cfg.ClientTLSInfo.Empty() {
			return nil, fmt.Errorf("TLS key/cert (--cert-file, --key-file) must be provided for client url %s with HTTPs scheme", u.String())
		}

		proto := "tcp"
		addr := u.Host
		if u.Scheme == "unix" || u.Scheme == "unixs" {
			proto = "unix"
			addr = u.Host + u.Path
		}

		sctx.secure = u.Scheme == "https" || u.Scheme == "unixs"
		sctx.insecure = !sctx.secure
		if oldctx := sctxs[addr]; oldctx != nil {
			oldctx.secure = oldctx.secure || sctx.secure
			oldctx.insecure = oldctx.insecure || sctx.insecure
			continue
		}

		if sctx.l, err = net.Listen(proto, addr); err != nil {
			return nil, err
		}
		// net.Listener will rewrite ipv4 0.0.0.0 to ipv6 [::], breaking
		// hosts that disable ipv6. So, use the address given by the user.
		sctx.addr = addr

		if fdLimit, fderr := runtimeutil.FDLimit(); fderr == nil {
			if fdLimit <= reservedInternalFDNum {
				plog.Fatalf("file descriptor limit[%d] of etcd process is too low, and should be set higher than %d to ensure internal usage", fdLimit, reservedInternalFDNum)
			}
			sctx.l = transport.LimitListener(sctx.l, int(fdLimit-reservedInternalFDNum))
		}

		if proto == "tcp" {
			if sctx.l, err = transport.NewKeepAliveListener(sctx.l, "tcp", nil); err != nil {
				return nil, err
			}
		}

		plog.Info("listening for client requests on ", u.Host)
		defer func() {
			if err != nil {
				sctx.l.Close()
				plog.Info("stopping listening for client requests on ", u.Host)
			}
		}()
		for k := range cfg.UserHandlers {
			sctx.userHandlers[k] = cfg.UserHandlers[k]
		}
		sctx.serviceRegister = cfg.ServiceRegister
		if cfg.EnablePprof || cfg.Debug {
			sctx.registerPprof()
		}
		if cfg.Debug {
			sctx.registerTrace()
		}
		sctxs[addr] = sctx
	}
	return sctxs, nil
}

func (e *Etcd) serve() (err error) {
	var ctlscfg *tls.Config
	if !e.cfg.ClientTLSInfo.Empty() {
		plog.Infof("ClientTLS: %s", e.cfg.ClientTLSInfo)
		if ctlscfg, err = e.cfg.ClientTLSInfo.ServerConfig(); err != nil {
			return err
		}
	}

	if e.cfg.CorsInfo.String() != "" {
		plog.Infof("cors = %s", e.cfg.CorsInfo)
	}

	// Start the peer server in a goroutine
	for _, pl := range e.Peers {
		go func(l *peerListener) {
			e.errHandler(l.serve())
		}(pl)
	}

	// Start a client server goroutine for each listen address
	var h http.Handler
	if e.Config().EnableV2 {
		h = v2http.NewClientHandler(e.Server, e.Server.Cfg.ReqTimeout())
	} else {
		mux := http.NewServeMux()
		etcdhttp.HandleBasic(mux, e.Server)
		h = mux
	}
	h = http.Handler(&cors.CORSHandler{Handler: h, Info: e.cfg.CorsInfo})

	gopts := []grpc.ServerOption{}
	if e.cfg.GRPCKeepAliveMinTime > time.Duration(0) {
		gopts = append(gopts, grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             e.cfg.GRPCKeepAliveMinTime,
			PermitWithoutStream: false,
		}))
	}
	if e.cfg.GRPCKeepAliveInterval > time.Duration(0) &&
		e.cfg.GRPCKeepAliveTimeout > time.Duration(0) {
		gopts = append(gopts, grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    e.cfg.GRPCKeepAliveInterval,
			Timeout: e.cfg.GRPCKeepAliveTimeout,
		}))
	}
	for _, sctx := range e.sctxs {
		go func(s *serveCtx) {
			e.errHandler(s.serve(e.Server, ctlscfg, h, e.errHandler, gopts...))
		}(sctx)
	}
	return nil
}

func (e *Etcd) errHandler(err error) {
	select {
	case <-e.stopc:
		return
	default:
	}
	select {
	case <-e.stopc:
	case e.errc <- err:
	}
}
