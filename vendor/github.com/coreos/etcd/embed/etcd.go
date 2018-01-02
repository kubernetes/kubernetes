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
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"path/filepath"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/pkg/cors"
	runtimeutil "github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/rafthttp"
	"github.com/coreos/pkg/capnslog"
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
	Peers   []net.Listener
	Clients []net.Listener
	Server  *etcdserver.EtcdServer

	cfg   Config
	errc  chan error
	sctxs map[string]*serveCtx
}

// StartEtcd launches the etcd server and HTTP handlers for client/server communication.
// The returned Etcd.Server is not guaranteed to have joined the cluster. Wait
// on the Etcd.Server.ReadyNotify() channel to know when it completes and is ready for use.
func StartEtcd(inCfg *Config) (e *Etcd, err error) {
	if err = inCfg.Validate(); err != nil {
		return nil, err
	}
	e = &Etcd{cfg: *inCfg}
	cfg := &e.cfg
	defer func() {
		if e != nil && err != nil {
			e.Close()
			e = nil
		}
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
		StrictReconfigCheck:     cfg.StrictReconfigCheck,
		ClientCertAuthEnabled:   cfg.ClientTLSInfo.ClientCertAuth,
	}

	if e.Server, err = etcdserver.NewServer(srvcfg); err != nil {
		return
	}

	// buffer channel so goroutines on closed connections won't wait forever
	e.errc = make(chan error, len(e.Peers)+len(e.Clients)+2*len(e.sctxs))

	e.Server.Start()
	if err = e.serve(); err != nil {
		return
	}
	return
}

// Config returns the current configuration.
func (e *Etcd) Config() Config {
	return e.cfg
}

func (e *Etcd) Close() {
	for _, sctx := range e.sctxs {
		sctx.cancel()
	}
	for i := range e.Peers {
		if e.Peers[i] != nil {
			e.Peers[i].Close()
		}
	}
	for i := range e.Clients {
		if e.Clients[i] != nil {
			e.Clients[i].Close()
		}
	}
	if e.Server != nil {
		e.Server.Stop()
	}
}

func (e *Etcd) Err() <-chan error { return e.errc }

func startPeerListeners(cfg *Config) (plns []net.Listener, err error) {
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

	plns = make([]net.Listener, len(cfg.LPUrls))
	defer func() {
		if err == nil {
			return
		}
		for i := range plns {
			if plns[i] == nil {
				continue
			}
			plns[i].Close()
			plog.Info("stopping listening for peers on ", cfg.LPUrls[i].String())
		}
	}()

	for i, u := range cfg.LPUrls {
		var tlscfg *tls.Config
		if u.Scheme == "http" {
			if !cfg.PeerTLSInfo.Empty() {
				plog.Warningf("The scheme of peer url %s is HTTP while peer key/cert files are presented. Ignored peer key/cert files.", u.String())
			}
			if cfg.PeerTLSInfo.ClientCertAuth {
				plog.Warningf("The scheme of peer url %s is HTTP while client cert auth (--peer-client-cert-auth) is enabled. Ignored client cert auth for this url.", u.String())
			}
		}
		if !cfg.PeerTLSInfo.Empty() {
			if tlscfg, err = cfg.PeerTLSInfo.ServerConfig(); err != nil {
				return nil, err
			}
		}
		if plns[i], err = rafthttp.NewListener(u, tlscfg); err != nil {
			return nil, err
		}
		plog.Info("listening for peers on ", u.String())
	}
	return plns, nil
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
		plog.Infof("pprof is enabled under %s", pprofPrefix)
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
		if u.Scheme == "unix" || u.Scheme == "unixs" {
			proto = "unix"
		}

		sctx.secure = u.Scheme == "https" || u.Scheme == "unixs"
		sctx.insecure = !sctx.secure
		if oldctx := sctxs[u.Host]; oldctx != nil {
			oldctx.secure = oldctx.secure || sctx.secure
			oldctx.insecure = oldctx.insecure || sctx.insecure
			continue
		}

		if sctx.l, err = net.Listen(proto, u.Host); err != nil {
			return nil, err
		}

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
		if cfg.EnablePprof {
			sctx.registerPprof()
		}
		sctxs[u.Host] = sctx
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
	ph := v2http.NewPeerHandler(e.Server)
	for _, l := range e.Peers {
		go func(l net.Listener) {
			e.errc <- servePeerHTTP(l, ph)
		}(l)
	}

	// Start a client server goroutine for each listen address
	ch := http.Handler(&cors.CORSHandler{
		Handler: v2http.NewClientHandler(e.Server, e.Server.Cfg.ReqTimeout()),
		Info:    e.cfg.CorsInfo,
	})
	for _, sctx := range e.sctxs {
		// read timeout does not work with http close notify
		// TODO: https://github.com/golang/go/issues/9524
		go func(s *serveCtx) {
			e.errc <- s.serve(e.Server, ctlscfg, ch, e.errc)
		}(sctx)
	}
	return nil
}
