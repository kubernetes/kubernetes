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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/coreos/etcd/discovery"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/fileutil"
	pkgioutil "github.com/coreos/etcd/pkg/ioutil"
	"github.com/coreos/etcd/pkg/osutil"
	runtimeutil "github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/proxy/httpproxy"
	"github.com/coreos/etcd/rafthttp"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-systemd/daemon"
	systemdutil "github.com/coreos/go-systemd/util"
	"github.com/coreos/pkg/capnslog"
	"github.com/prometheus/client_golang/prometheus"
)

type dirType string

var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcdmain")

const (
	// the owner can make/remove files inside the directory
	privateDirMode = 0700

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

var (
	dirMember = dirType("member")
	dirProxy  = dirType("proxy")
	dirEmpty  = dirType("empty")
)

func startEtcdOrProxyV2() {
	cfg := NewConfig()
	err := cfg.Parse(os.Args[1:])
	if err != nil {
		plog.Errorf("error verifying flags, %v. See 'etcd --help'.", err)
		switch err {
		case errUnsetAdvertiseClientURLsFlag:
			plog.Errorf("When listening on specific address(es), this etcd process must advertise accessible url(s) to each connected client.")
		}
		os.Exit(1)
	}
	setupLogging(cfg)

	var stopped <-chan struct{}

	plog.Infof("etcd Version: %s\n", version.Version)
	plog.Infof("Git SHA: %s\n", version.GitSHA)
	plog.Infof("Go Version: %s\n", runtime.Version())
	plog.Infof("Go OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	GoMaxProcs := runtime.GOMAXPROCS(0)
	plog.Infof("setting maximum number of CPUs to %d, total number of available CPUs is %d", GoMaxProcs, runtime.NumCPU())

	// TODO: check whether fields are set instead of whether fields have default value
	if cfg.Name != defaultName && cfg.InitialCluster == initialClusterFromName(defaultName) {
		cfg.InitialCluster = initialClusterFromName(cfg.Name)
	}

	if cfg.Dir == "" {
		cfg.Dir = fmt.Sprintf("%v.etcd", cfg.Name)
		plog.Warningf("no data-dir provided, using default data-dir ./%s", cfg.Dir)
	}

	which := identifyDataDirOrDie(cfg.Dir)
	if which != dirEmpty {
		plog.Noticef("the server is already initialized as %v before, starting as etcd %v...", which, which)
		switch which {
		case dirMember:
			stopped, err = startEtcd(cfg)
		case dirProxy:
			err = startProxy(cfg)
		default:
			plog.Panicf("unhandled dir type %v", which)
		}
	} else {
		shouldProxy := cfg.isProxy()
		if !shouldProxy {
			stopped, err = startEtcd(cfg)
			if derr, ok := err.(*etcdserver.DiscoveryError); ok && derr.Err == discovery.ErrFullCluster {
				if cfg.shouldFallbackToProxy() {
					plog.Noticef("discovery cluster full, falling back to %s", fallbackFlagProxy)
					shouldProxy = true
				}
			}
		}
		if shouldProxy {
			err = startProxy(cfg)
		}
	}

	if err != nil {
		if derr, ok := err.(*etcdserver.DiscoveryError); ok {
			switch derr.Err {
			case discovery.ErrDuplicateID:
				plog.Errorf("member %q has previously registered with discovery service token (%s).", cfg.Name, cfg.Durl)
				plog.Errorf("But etcd could not find valid cluster configuration in the given data dir (%s).", cfg.Dir)
				plog.Infof("Please check the given data dir path if the previous bootstrap succeeded")
				plog.Infof("or use a new discovery token if the previous bootstrap failed.")
			case discovery.ErrDuplicateName:
				plog.Errorf("member with duplicated name has registered with discovery service token(%s).", cfg.Durl)
				plog.Errorf("please check (cURL) the discovery token for more information.")
				plog.Errorf("please do not reuse the discovery token and generate a new one to bootstrap the cluster.")
			default:
				plog.Errorf("%v", err)
				plog.Infof("discovery token %s was used, but failed to bootstrap the cluster.", cfg.Durl)
				plog.Infof("please generate a new discovery token and try to bootstrap again.")
			}
			os.Exit(1)
		}

		if strings.Contains(err.Error(), "include") && strings.Contains(err.Error(), "--initial-cluster") {
			plog.Infof("%v", err)
			if cfg.InitialCluster == initialClusterFromName(cfg.Name) {
				plog.Infof("forgot to set --initial-cluster flag?")
			}
			if types.URLs(cfg.apurls).String() == defaultInitialAdvertisePeerURLs {
				plog.Infof("forgot to set --initial-advertise-peer-urls flag?")
			}
			if cfg.InitialCluster == initialClusterFromName(cfg.Name) && len(cfg.Durl) == 0 {
				plog.Infof("if you want to use discovery service, please set --discovery flag.")
			}
			os.Exit(1)
		}
		plog.Fatalf("%v", err)
	}

	osutil.HandleInterrupts()

	if systemdutil.IsRunningSystemd() {
		// At this point, the initialization of etcd is done.
		// The listeners are listening on the TCP ports and ready
		// for accepting connections. The etcd instance should be
		// joined with the cluster and ready to serve incoming
		// connections.
		sent, err := daemon.SdNotify("READY=1")
		if err != nil {
			plog.Errorf("failed to notify systemd for readiness: %v", err)
		}
		if !sent {
			plog.Errorf("forgot to set Type=notify in systemd service file?")
		}
	}

	<-stopped
	osutil.Exit(0)
}

// startEtcd launches the etcd server and HTTP handlers for client/server communication.
func startEtcd(cfg *config) (<-chan struct{}, error) {
	var (
		urlsmap types.URLsMap
		token   string
		err     error
	)
	if !isMemberInitialized(cfg) {
		urlsmap, token, err = getPeerURLsMapAndToken(cfg, "etcd")
		if err != nil {
			return nil, fmt.Errorf("error setting up initial cluster: %v", err)
		}
	}

	if cfg.PeerAutoTLS && cfg.peerTLSInfo.Empty() {
		var phosts []string
		for _, u := range cfg.lpurls {
			phosts = append(phosts, u.Host)
		}
		cfg.peerTLSInfo, err = transport.SelfCert(path.Join(cfg.Dir, "fixtures/peer"), phosts)
		if err != nil {
			plog.Fatalf("could not get certs (%v)", err)
		}
	} else if cfg.PeerAutoTLS {
		plog.Warningf("ignoring peer auto TLS since certs given")
	}

	if !cfg.peerTLSInfo.Empty() {
		plog.Infof("peerTLS: %s", cfg.peerTLSInfo)
	}

	var plns []net.Listener
	for _, u := range cfg.lpurls {
		if u.Scheme == "http" {
			if !cfg.peerTLSInfo.Empty() {
				plog.Warningf("The scheme of peer url %s is HTTP while peer key/cert files are presented. Ignored peer key/cert files.", u.String())
			}
			if cfg.peerTLSInfo.ClientCertAuth {
				plog.Warningf("The scheme of peer url %s is HTTP while client cert auth (--peer-client-cert-auth) is enabled. Ignored client cert auth for this url.", u.String())
			}
		}
		var (
			l      net.Listener
			tlscfg *tls.Config
		)

		if !cfg.peerTLSInfo.Empty() {
			tlscfg, err = cfg.peerTLSInfo.ServerConfig()
			if err != nil {
				return nil, err
			}
		}

		l, err = rafthttp.NewListener(u, tlscfg)
		if err != nil {
			return nil, err
		}

		urlStr := u.String()
		plog.Info("listening for peers on ", urlStr)
		defer func() {
			if err != nil {
				l.Close()
				plog.Info("stopping listening for peers on ", urlStr)
			}
		}()
		plns = append(plns, l)
	}

	if cfg.ClientAutoTLS && cfg.clientTLSInfo.Empty() {
		var chosts []string
		for _, u := range cfg.lcurls {
			chosts = append(chosts, u.Host)
		}
		cfg.clientTLSInfo, err = transport.SelfCert(path.Join(cfg.Dir, "fixtures/client"), chosts)
		if err != nil {
			plog.Fatalf("could not get certs (%v)", err)
		}
	} else if cfg.ClientAutoTLS {
		plog.Warningf("ignoring client auto TLS since certs given")
	}

	var ctlscfg *tls.Config
	if !cfg.clientTLSInfo.Empty() {
		plog.Infof("clientTLS: %s", cfg.clientTLSInfo)
		ctlscfg, err = cfg.clientTLSInfo.ServerConfig()
		if err != nil {
			return nil, err
		}
	}

	sctxs := make(map[string]*serveCtx)
	for _, u := range cfg.lcurls {
		if u.Scheme == "http" {
			if !cfg.clientTLSInfo.Empty() {
				plog.Warningf("The scheme of client url %s is HTTP while peer key/cert files are presented. Ignored key/cert files.", u.String())
			}
			if cfg.clientTLSInfo.ClientCertAuth {
				plog.Warningf("The scheme of client url %s is HTTP while client cert auth (--client-cert-auth) is enabled. Ignored client cert auth for this url.", u.String())
			}
		}
		if u.Scheme == "https" && ctlscfg == nil {
			return nil, fmt.Errorf("TLS key/cert (--cert-file, --key-file) must be provided for client url %s with HTTPs scheme", u.String())
		}

		ctx := &serveCtx{host: u.Host}

		if u.Scheme == "https" {
			ctx.secure = true
		} else {
			ctx.insecure = true
		}

		if sctxs[u.Host] != nil {
			if ctx.secure {
				sctxs[u.Host].secure = true
			}
			if ctx.insecure {
				sctxs[u.Host].insecure = true
			}
			continue
		}

		var l net.Listener

		l, err = net.Listen("tcp", u.Host)
		if err != nil {
			return nil, err
		}

		var fdLimit uint64
		if fdLimit, err = runtimeutil.FDLimit(); err == nil {
			if fdLimit <= reservedInternalFDNum {
				plog.Fatalf("file descriptor limit[%d] of etcd process is too low, and should be set higher than %d to ensure internal usage", fdLimit, reservedInternalFDNum)
			}
			l = transport.LimitListener(l, int(fdLimit-reservedInternalFDNum))
		}

		l, err = transport.NewKeepAliveListener(l, "tcp", nil)
		ctx.l = l
		if err != nil {
			return nil, err
		}

		plog.Info("listening for client requests on ", u.Host)
		defer func() {
			if err != nil {
				l.Close()
				plog.Info("stopping listening for client requests on ", u.Host)
			}
		}()
		sctxs[u.Host] = ctx
	}

	srvcfg := &etcdserver.ServerConfig{
		Name:                    cfg.Name,
		ClientURLs:              cfg.acurls,
		PeerURLs:                cfg.apurls,
		DataDir:                 cfg.Dir,
		DedicatedWALDir:         cfg.WalDir,
		SnapCount:               cfg.SnapCount,
		MaxSnapFiles:            cfg.MaxSnapFiles,
		MaxWALFiles:             cfg.MaxWalFiles,
		InitialPeerURLsMap:      urlsmap,
		InitialClusterToken:     token,
		DiscoveryURL:            cfg.Durl,
		DiscoveryProxy:          cfg.Dproxy,
		NewCluster:              cfg.isNewCluster(),
		ForceNewCluster:         cfg.ForceNewCluster,
		PeerTLSInfo:             cfg.peerTLSInfo,
		TickMs:                  cfg.TickMs,
		ElectionTicks:           cfg.electionTicks(),
		AutoCompactionRetention: cfg.autoCompactionRetention,
		QuotaBackendBytes:       cfg.QuotaBackendBytes,
		StrictReconfigCheck:     cfg.StrictReconfigCheck,
		EnablePprof:             cfg.enablePprof,
		ClientCertAuthEnabled:   cfg.clientTLSInfo.ClientCertAuth,
	}
	var s *etcdserver.EtcdServer
	s, err = etcdserver.NewServer(srvcfg)
	if err != nil {
		return nil, err
	}
	s.Start()
	osutil.RegisterInterruptHandler(s.Stop)

	if cfg.corsInfo.String() != "" {
		plog.Infof("cors = %s", cfg.corsInfo)
	}
	ch := http.Handler(&cors.CORSHandler{
		Handler: v2http.NewClientHandler(s, srvcfg.ReqTimeout()),
		Info:    cfg.corsInfo,
	})
	ph := v2http.NewPeerHandler(s)

	// Start the peer server in a goroutine
	for _, l := range plns {
		go func(l net.Listener) {
			plog.Fatal(servePeerHTTP(l, ph))
		}(l)
	}
	// Start a client server goroutine for each listen address
	for _, sctx := range sctxs {
		go func(sctx *serveCtx) {
			// read timeout does not work with http close notify
			// TODO: https://github.com/golang/go/issues/9524
			plog.Fatal(serve(sctx, s, ctlscfg, ch))
		}(sctx)
	}

	<-s.ReadyNotify()
	return s.StopNotify(), nil
}

// startProxy launches an HTTP proxy for client communication which proxies to other etcd nodes.
func startProxy(cfg *config) error {
	plog.Notice("proxy: this proxy supports v2 API only!")

	pt, err := transport.NewTimeoutTransport(cfg.peerTLSInfo, time.Duration(cfg.ProxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}
	pt.MaxIdleConnsPerHost = httpproxy.DefaultMaxIdleConnsPerHost

	tr, err := transport.NewTimeoutTransport(cfg.peerTLSInfo, time.Duration(cfg.ProxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}

	cfg.Dir = path.Join(cfg.Dir, "proxy")
	err = os.MkdirAll(cfg.Dir, privateDirMode)
	if err != nil {
		return err
	}

	var peerURLs []string
	clusterfile := path.Join(cfg.Dir, "cluster")

	b, err := ioutil.ReadFile(clusterfile)
	switch {
	case err == nil:
		if cfg.Durl != "" {
			plog.Warningf("discovery token ignored since the proxy has already been initialized. Valid cluster file found at %q", clusterfile)
		}
		if cfg.DnsCluster != "" {
			plog.Warningf("DNS SRV discovery ignored since the proxy has already been initialized. Valid cluster file found at %q", clusterfile)
		}
		urls := struct{ PeerURLs []string }{}
		err = json.Unmarshal(b, &urls)
		if err != nil {
			return err
		}
		peerURLs = urls.PeerURLs
		plog.Infof("proxy: using peer urls %v from cluster file %q", peerURLs, clusterfile)
	case os.IsNotExist(err):
		var urlsmap types.URLsMap
		urlsmap, _, err = getPeerURLsMapAndToken(cfg, "proxy")
		if err != nil {
			return fmt.Errorf("error setting up initial cluster: %v", err)
		}

		if cfg.Durl != "" {
			var s string
			s, err = discovery.GetCluster(cfg.Durl, cfg.Dproxy)
			if err != nil {
				return err
			}
			if urlsmap, err = types.NewURLsMap(s); err != nil {
				return err
			}
		}
		peerURLs = urlsmap.URLs()
		plog.Infof("proxy: using peer urls %v ", peerURLs)
	default:
		return err
	}

	clientURLs := []string{}
	uf := func() []string {
		gcls, gerr := etcdserver.GetClusterFromRemotePeers(peerURLs, tr)
		// TODO: remove the 2nd check when we fix GetClusterFromRemotePeers
		// GetClusterFromRemotePeers should not return nil error with an invalid empty cluster
		if gerr != nil {
			plog.Warningf("proxy: %v", gerr)
			return []string{}
		}
		if len(gcls.Members()) == 0 {
			return clientURLs
		}
		clientURLs = gcls.ClientURLs()

		urls := struct{ PeerURLs []string }{gcls.PeerURLs()}
		b, jerr := json.Marshal(urls)
		if jerr != nil {
			plog.Warningf("proxy: error on marshal peer urls %s", jerr)
			return clientURLs
		}

		err = pkgioutil.WriteAndSyncFile(clusterfile+".bak", b, 0600)
		if err != nil {
			plog.Warningf("proxy: error on writing urls %s", err)
			return clientURLs
		}
		err = os.Rename(clusterfile+".bak", clusterfile)
		if err != nil {
			plog.Warningf("proxy: error on updating clusterfile %s", err)
			return clientURLs
		}
		if !reflect.DeepEqual(gcls.PeerURLs(), peerURLs) {
			plog.Noticef("proxy: updated peer urls in cluster file from %v to %v", peerURLs, gcls.PeerURLs())
		}
		peerURLs = gcls.PeerURLs()

		return clientURLs
	}
	ph := httpproxy.NewHandler(pt, uf, time.Duration(cfg.ProxyFailureWaitMs)*time.Millisecond, time.Duration(cfg.ProxyRefreshIntervalMs)*time.Millisecond)
	ph = &cors.CORSHandler{
		Handler: ph,
		Info:    cfg.corsInfo,
	}

	if cfg.isReadonlyProxy() {
		ph = httpproxy.NewReadonlyHandler(ph)
	}
	// Start a proxy server goroutine for each listen address
	for _, u := range cfg.lcurls {
		var (
			l      net.Listener
			tlscfg *tls.Config
		)
		if !cfg.clientTLSInfo.Empty() {
			tlscfg, err = cfg.clientTLSInfo.ServerConfig()
			if err != nil {
				return err
			}
		}

		l, err := transport.NewListener(u.Host, u.Scheme, tlscfg)
		if err != nil {
			return err
		}

		host := u.String()
		go func() {
			plog.Info("proxy: listening for client requests on ", host)
			mux := http.NewServeMux()
			mux.Handle("/metrics", prometheus.Handler())
			mux.Handle("/", ph)
			plog.Fatal(http.Serve(l, mux))
		}()
	}
	return nil
}

// getPeerURLsMapAndToken sets up an initial peer URLsMap and cluster token for bootstrap or discovery.
func getPeerURLsMapAndToken(cfg *config, which string) (urlsmap types.URLsMap, token string, err error) {
	switch {
	case cfg.Durl != "":
		urlsmap = types.URLsMap{}
		// If using discovery, generate a temporary cluster based on
		// self's advertised peer URLs
		urlsmap[cfg.Name] = cfg.apurls
		token = cfg.Durl
	case cfg.DnsCluster != "":
		var clusterStr string
		clusterStr, token, err = discovery.SRVGetCluster(cfg.Name, cfg.DnsCluster, cfg.InitialClusterToken, cfg.apurls)
		if err != nil {
			return nil, "", err
		}
		if strings.Contains(clusterStr, "https://") && cfg.peerTLSInfo.CAFile == "" {
			cfg.peerTLSInfo.ServerName = cfg.DnsCluster
		}
		urlsmap, err = types.NewURLsMap(clusterStr)
		// only etcd member must belong to the discovered cluster.
		// proxy does not need to belong to the discovered cluster.
		if which == "etcd" {
			if _, ok := urlsmap[cfg.Name]; !ok {
				return nil, "", fmt.Errorf("cannot find local etcd member %q in SRV records", cfg.Name)
			}
		}
	default:
		// We're statically configured, and cluster has appropriately been set.
		urlsmap, err = types.NewURLsMap(cfg.InitialCluster)
		token = cfg.InitialClusterToken
	}
	return urlsmap, token, err
}

// identifyDataDirOrDie returns the type of the data dir.
// Dies if the datadir is invalid.
func identifyDataDirOrDie(dir string) dirType {
	names, err := fileutil.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return dirEmpty
		}
		plog.Fatalf("error listing data dir: %s", dir)
	}

	var m, p bool
	for _, name := range names {
		switch dirType(name) {
		case dirMember:
			m = true
		case dirProxy:
			p = true
		default:
			plog.Warningf("found invalid file/dir %s under data dir %s (Ignore this if you are upgrading etcd)", name, dir)
		}
	}

	if m && p {
		plog.Fatal("invalid datadir. Both member and proxy directories exist.")
	}
	if m {
		return dirMember
	}
	if p {
		return dirProxy
	}
	return dirEmpty
}

func setupLogging(cfg *config) {
	capnslog.SetGlobalLogLevel(capnslog.INFO)
	if cfg.Debug {
		capnslog.SetGlobalLogLevel(capnslog.DEBUG)
	}
	if cfg.LogPkgLevels != "" {
		repoLog := capnslog.MustRepoLogger("github.com/coreos/etcd")
		settings, err := repoLog.ParseLogLevelConfig(cfg.LogPkgLevels)
		if err != nil {
			plog.Warningf("couldn't parse log level string: %s, continuing with default levels", err.Error())
			return
		}
		repoLog.SetLogLevel(settings)
	}
}

func checkSupportArch() {
	// TODO qualify arm64
	if runtime.GOARCH == "amd64" {
		return
	}
	if env, ok := os.LookupEnv("ETCD_UNSUPPORTED_ARCH"); ok && env == runtime.GOARCH {
		plog.Warningf("running etcd on unsupported architecture %q since ETCD_UNSUPPORTED_ARCH is set", env)
		return
	}
	plog.Errorf("etcd on unsupported platform without ETCD_UNSUPPORTED_ARCH=%s set.", runtime.GOARCH)
	os.Exit(1)
}
