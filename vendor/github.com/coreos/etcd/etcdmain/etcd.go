// Copyright 2015 CoreOS, Inc.
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

// TODO: support arm64
// +build amd64

package etcdmain

import (
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
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	"github.com/coreos/etcd/etcdserver/etcdhttp"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/fileutil"
	pkgioutil "github.com/coreos/etcd/pkg/ioutil"
	"github.com/coreos/etcd/pkg/osutil"
	runtimeutil "github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/proxy"
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

func Main() {
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
	if cfg.name != defaultName && cfg.initialCluster == initialClusterFromName(defaultName) {
		cfg.initialCluster = initialClusterFromName(cfg.name)
	}

	if cfg.dir == "" {
		cfg.dir = fmt.Sprintf("%v.etcd", cfg.name)
		plog.Warningf("no data-dir provided, using default data-dir ./%s", cfg.dir)
	}

	which := identifyDataDirOrDie(cfg.dir)
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
				plog.Errorf("member %q has previously registered with discovery service token (%s).", cfg.name, cfg.durl)
				plog.Errorf("But etcd could not find valid cluster configuration in the given data dir (%s).", cfg.dir)
				plog.Infof("Please check the given data dir path if the previous bootstrap succeeded")
				plog.Infof("or use a new discovery token if the previous bootstrap failed.")
			case discovery.ErrDuplicateName:
				plog.Errorf("member with duplicated name has registered with discovery service token(%s).", cfg.durl)
				plog.Errorf("please check (cURL) the discovery token for more information.")
				plog.Errorf("please do not reuse the discovery token and generate a new one to bootstrap the cluster.")
			default:
				plog.Errorf("%v", err)
				plog.Infof("discovery token %s was used, but failed to bootstrap the cluster.", cfg.durl)
				plog.Infof("please generate a new discovery token and try to bootstrap again.")
			}
			os.Exit(1)
		}

		if strings.Contains(err.Error(), "include") && strings.Contains(err.Error(), "--initial-cluster") {
			plog.Infof("%v", err)
			if cfg.initialCluster == initialClusterFromName(cfg.name) {
				plog.Infof("forgot to set --initial-cluster flag?")
			}
			if types.URLs(cfg.apurls).String() == defaultInitialAdvertisePeerURLs {
				plog.Infof("forgot to set --initial-advertise-peer-urls flag?")
			}
			if cfg.initialCluster == initialClusterFromName(cfg.name) && len(cfg.durl) == 0 {
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
		// for accepting connections.
		// The http server is probably ready for serving incoming
		// connections. If it is not, the connection might be pending
		// for less than one second.
		err := daemon.SdNotify("READY=1")
		if err != nil {
			plog.Errorf("failed to notify systemd for readiness: %v", err)
			if err == daemon.SdNotifyNoSocket {
				plog.Errorf("forgot to set Type=notify in systemd service file?")
			}
		}
	}

	<-stopped
	osutil.Exit(0)
}

// startEtcd launches the etcd server and HTTP handlers for client/server communication.
func startEtcd(cfg *config) (<-chan struct{}, error) {
	urlsmap, token, err := getPeerURLsMapAndToken(cfg, "etcd")
	if err != nil {
		return nil, fmt.Errorf("error setting up initial cluster: %v", err)
	}

	if !cfg.peerTLSInfo.Empty() {
		plog.Infof("peerTLS: %s", cfg.peerTLSInfo)
	}
	plns := make([]net.Listener, 0)
	for _, u := range cfg.lpurls {
		if u.Scheme == "http" && !cfg.peerTLSInfo.Empty() {
			plog.Warningf("The scheme of peer url %s is http while peer key/cert files are presented. Ignored peer key/cert files.", u.String())
		}
		var l net.Listener
		l, err = rafthttp.NewListener(u, cfg.peerTLSInfo)
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

	if !cfg.clientTLSInfo.Empty() {
		plog.Infof("clientTLS: %s", cfg.clientTLSInfo)
	}
	clns := make([]net.Listener, 0)
	for _, u := range cfg.lcurls {
		if u.Scheme == "http" && !cfg.clientTLSInfo.Empty() {
			plog.Warningf("The scheme of client url %s is http while client key/cert files are presented. Ignored client key/cert files.", u.String())
		}
		var l net.Listener
		l, err = net.Listen("tcp", u.Host)
		if err != nil {
			return nil, err
		}

		if fdLimit, err := runtimeutil.FDLimit(); err == nil {
			if fdLimit <= reservedInternalFDNum {
				plog.Fatalf("file descriptor limit[%d] of etcd process is too low, and should be set higher than %d to ensure internal usage", fdLimit, reservedInternalFDNum)
			}
			l = transport.LimitListener(l, int(fdLimit-reservedInternalFDNum))
		}

		// Do not wrap around this listener if TLS Info is set.
		// HTTPS server expects TLS Conn created by TLSListener.
		l, err = transport.NewKeepAliveListener(l, u.Scheme, cfg.clientTLSInfo)
		if err != nil {
			return nil, err
		}

		urlStr := u.String()
		plog.Info("listening for client requests on ", urlStr)
		defer func() {
			if err != nil {
				l.Close()
				plog.Info("stopping listening for client requests on ", urlStr)
			}
		}()
		clns = append(clns, l)
	}

	var v3l net.Listener
	if cfg.v3demo {
		v3l, err = net.Listen("tcp", cfg.gRPCAddr)
		if err != nil {
			plog.Fatal(err)
		}
		plog.Infof("listening for client rpc on %s", cfg.gRPCAddr)
	}

	srvcfg := &etcdserver.ServerConfig{
		Name:                    cfg.name,
		ClientURLs:              cfg.acurls,
		PeerURLs:                cfg.apurls,
		DataDir:                 cfg.dir,
		DedicatedWALDir:         cfg.walDir,
		SnapCount:               cfg.snapCount,
		MaxSnapFiles:            cfg.maxSnapFiles,
		MaxWALFiles:             cfg.maxWalFiles,
		InitialPeerURLsMap:      urlsmap,
		InitialClusterToken:     token,
		DiscoveryURL:            cfg.durl,
		DiscoveryProxy:          cfg.dproxy,
		NewCluster:              cfg.isNewCluster(),
		ForceNewCluster:         cfg.forceNewCluster,
		PeerTLSInfo:             cfg.peerTLSInfo,
		TickMs:                  cfg.TickMs,
		ElectionTicks:           cfg.electionTicks(),
		V3demo:                  cfg.v3demo,
		AutoCompactionRetention: cfg.autoCompactionRetention,
		StrictReconfigCheck:     cfg.strictReconfigCheck,
		EnablePprof:             cfg.enablePprof,
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
	ch := &cors.CORSHandler{
		Handler: etcdhttp.NewClientHandler(s, srvcfg.ReqTimeout()),
		Info:    cfg.corsInfo,
	}
	ph := etcdhttp.NewPeerHandler(s)
	// Start the peer server in a goroutine
	for _, l := range plns {
		go func(l net.Listener) {
			plog.Fatal(serveHTTP(l, ph, 5*time.Minute))
		}(l)
	}
	// Start a client server goroutine for each listen address
	for _, l := range clns {
		go func(l net.Listener) {
			// read timeout does not work with http close notify
			// TODO: https://github.com/golang/go/issues/9524
			plog.Fatal(serveHTTP(l, ch, 0))
		}(l)
	}

	if cfg.v3demo {
		// set up v3 demo rpc
		tls := &cfg.clientTLSInfo
		if cfg.clientTLSInfo.Empty() {
			tls = nil
		}
		grpcServer, err := v3rpc.Server(s, tls)
		if err != nil {
			s.Stop()
			<-s.StopNotify()
			return nil, err
		}
		go func() { plog.Fatal(grpcServer.Serve(v3l)) }()
	}

	return s.StopNotify(), nil
}

// startProxy launches an HTTP proxy for client communication which proxies to other etcd nodes.
func startProxy(cfg *config) error {
	pt, err := transport.NewTimeoutTransport(cfg.peerTLSInfo, time.Duration(cfg.proxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.proxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.proxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}
	pt.MaxIdleConnsPerHost = proxy.DefaultMaxIdleConnsPerHost

	tr, err := transport.NewTimeoutTransport(cfg.peerTLSInfo, time.Duration(cfg.proxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.proxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.proxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}

	cfg.dir = path.Join(cfg.dir, "proxy")
	err = os.MkdirAll(cfg.dir, 0700)
	if err != nil {
		return err
	}

	var peerURLs []string
	clusterfile := path.Join(cfg.dir, "cluster")

	b, err := ioutil.ReadFile(clusterfile)
	switch {
	case err == nil:
		if cfg.durl != "" {
			plog.Warningf("discovery token ignored since the proxy has already been initialized. Valid cluster file found at %q", clusterfile)
		}
		if cfg.dnsCluster != "" {
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
		urlsmap, _, err := getPeerURLsMapAndToken(cfg, "proxy")
		if err != nil {
			return fmt.Errorf("error setting up initial cluster: %v", err)
		}

		if cfg.durl != "" {
			s, err := discovery.GetCluster(cfg.durl, cfg.dproxy)
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
		gcls, err := etcdserver.GetClusterFromRemotePeers(peerURLs, tr)
		// TODO: remove the 2nd check when we fix GetClusterFromRemotePeers
		// GetClusterFromRemotePeers should not return nil error with an invalid empty cluster
		if err != nil {
			plog.Warningf("proxy: %v", err)
			return []string{}
		}
		if len(gcls.Members()) == 0 {
			return clientURLs
		}
		clientURLs = gcls.ClientURLs()

		urls := struct{ PeerURLs []string }{gcls.PeerURLs()}
		b, err := json.Marshal(urls)
		if err != nil {
			plog.Warningf("proxy: error on marshal peer urls %s", err)
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
	ph := proxy.NewHandler(pt, uf, time.Duration(cfg.proxyFailureWaitMs)*time.Millisecond, time.Duration(cfg.proxyRefreshIntervalMs)*time.Millisecond)
	ph = &cors.CORSHandler{
		Handler: ph,
		Info:    cfg.corsInfo,
	}

	if cfg.isReadonlyProxy() {
		ph = proxy.NewReadonlyHandler(ph)
	}
	// Start a proxy server goroutine for each listen address
	for _, u := range cfg.lcurls {
		l, err := transport.NewListener(u.Host, u.Scheme, cfg.clientTLSInfo)
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
	case cfg.durl != "":
		urlsmap = types.URLsMap{}
		// If using discovery, generate a temporary cluster based on
		// self's advertised peer URLs
		urlsmap[cfg.name] = cfg.apurls
		token = cfg.durl
	case cfg.dnsCluster != "":
		var clusterStr string
		clusterStr, token, err = discovery.SRVGetCluster(cfg.name, cfg.dnsCluster, cfg.initialClusterToken, cfg.apurls)
		if err != nil {
			return nil, "", err
		}
		urlsmap, err = types.NewURLsMap(clusterStr)
		// only etcd member must belong to the discovered cluster.
		// proxy does not need to belong to the discovered cluster.
		if which == "etcd" {
			if _, ok := urlsmap[cfg.name]; !ok {
				return nil, "", fmt.Errorf("cannot find local etcd member %q in SRV records", cfg.name)
			}
		}
	default:
		// We're statically configured, and cluster has appropriately been set.
		urlsmap, err = types.NewURLsMap(cfg.initialCluster)
		token = cfg.initialClusterToken
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
	if cfg.debug {
		capnslog.SetGlobalLogLevel(capnslog.DEBUG)
	}
	if cfg.logPkgLevels != "" {
		repoLog := capnslog.MustRepoLogger("github.com/coreos/etcd")
		settings, err := repoLog.ParseLogLevelConfig(cfg.logPkgLevels)
		if err != nil {
			plog.Warningf("couldn't parse log level string: %s, continuing with default levels", err.Error())
			return
		}
		repoLog.SetLogLevel(settings)
	}
}
