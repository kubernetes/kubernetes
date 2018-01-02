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
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/coreos/etcd/discovery"
	"github.com/coreos/etcd/embed"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/fileutil"
	pkgioutil "github.com/coreos/etcd/pkg/ioutil"
	"github.com/coreos/etcd/pkg/osutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/proxy/httpproxy"
	"github.com/coreos/etcd/version"
	"github.com/coreos/pkg/capnslog"
	"github.com/grpc-ecosystem/go-grpc-prometheus"
	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
)

type dirType string

var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcdmain")

var (
	dirMember = dirType("member")
	dirProxy  = dirType("proxy")
	dirEmpty  = dirType("empty")
)

func startEtcdOrProxyV2() {
	grpc.EnableTracing = false

	cfg := newConfig()
	defaultInitialCluster := cfg.InitialCluster

	err := cfg.parse(os.Args[1:])
	if err != nil {
		plog.Errorf("error verifying flags, %v. See 'etcd --help'.", err)
		switch err {
		case embed.ErrUnsetAdvertiseClientURLsFlag:
			plog.Errorf("When listening on specific address(es), this etcd process must advertise accessible url(s) to each connected client.")
		}
		os.Exit(1)
	}
	setupLogging(cfg)

	var stopped <-chan struct{}
	var errc <-chan error

	plog.Infof("etcd Version: %s\n", version.Version)
	plog.Infof("Git SHA: %s\n", version.GitSHA)
	plog.Infof("Go Version: %s\n", runtime.Version())
	plog.Infof("Go OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	GoMaxProcs := runtime.GOMAXPROCS(0)
	plog.Infof("setting maximum number of CPUs to %d, total number of available CPUs is %d", GoMaxProcs, runtime.NumCPU())

	defaultHost, dhErr := (&cfg.Config).UpdateDefaultClusterFromName(defaultInitialCluster)
	if defaultHost != "" {
		plog.Infof("advertising using detected default host %q", defaultHost)
	}
	if dhErr != nil {
		plog.Noticef("failed to detect default host (%v)", dhErr)
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
			stopped, errc, err = startEtcd(&cfg.Config)
		case dirProxy:
			err = startProxy(cfg)
		default:
			plog.Panicf("unhandled dir type %v", which)
		}
	} else {
		shouldProxy := cfg.isProxy()
		if !shouldProxy {
			stopped, errc, err = startEtcd(&cfg.Config)
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
			if cfg.InitialCluster == cfg.InitialClusterFromName(cfg.Name) {
				plog.Infof("forgot to set --initial-cluster flag?")
			}
			if types.URLs(cfg.APUrls).String() == embed.DefaultInitialAdvertisePeerURLs {
				plog.Infof("forgot to set --initial-advertise-peer-urls flag?")
			}
			if cfg.InitialCluster == cfg.InitialClusterFromName(cfg.Name) && len(cfg.Durl) == 0 {
				plog.Infof("if you want to use discovery service, please set --discovery flag.")
			}
			os.Exit(1)
		}
		plog.Fatalf("%v", err)
	}

	osutil.HandleInterrupts()

	// At this point, the initialization of etcd is done.
	// The listeners are listening on the TCP ports and ready
	// for accepting connections. The etcd instance should be
	// joined with the cluster and ready to serve incoming
	// connections.
	notifySystemd()

	select {
	case lerr := <-errc:
		// fatal out on listener errors
		plog.Fatal(lerr)
	case <-stopped:
	}

	osutil.Exit(0)
}

// startEtcd runs StartEtcd in addition to hooks needed for standalone etcd.
func startEtcd(cfg *embed.Config) (<-chan struct{}, <-chan error, error) {
	if cfg.Metrics == "extensive" {
		grpc_prometheus.EnableHandlingTimeHistogram()
	}

	e, err := embed.StartEtcd(cfg)
	if err != nil {
		return nil, nil, err
	}
	osutil.RegisterInterruptHandler(e.Server.Stop)
	select {
	case <-e.Server.ReadyNotify(): // wait for e.Server to join the cluster
	case <-e.Server.StopNotify(): // publish aborted from 'ErrStopped'
	}
	return e.Server.StopNotify(), e.Err(), nil
}

// startProxy launches an HTTP proxy for client communication which proxies to other etcd nodes.
func startProxy(cfg *config) error {
	plog.Notice("proxy: this proxy supports v2 API only!")

	pt, err := transport.NewTimeoutTransport(cfg.PeerTLSInfo, time.Duration(cfg.ProxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}
	pt.MaxIdleConnsPerHost = httpproxy.DefaultMaxIdleConnsPerHost

	tr, err := transport.NewTimeoutTransport(cfg.PeerTLSInfo, time.Duration(cfg.ProxyDialTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyReadTimeoutMs)*time.Millisecond, time.Duration(cfg.ProxyWriteTimeoutMs)*time.Millisecond)
	if err != nil {
		return err
	}

	cfg.Dir = filepath.Join(cfg.Dir, "proxy")
	err = os.MkdirAll(cfg.Dir, fileutil.PrivateDirMode)
	if err != nil {
		return err
	}

	var peerURLs []string
	clusterfile := filepath.Join(cfg.Dir, "cluster")

	b, err := ioutil.ReadFile(clusterfile)
	switch {
	case err == nil:
		if cfg.Durl != "" {
			plog.Warningf("discovery token ignored since the proxy has already been initialized. Valid cluster file found at %q", clusterfile)
		}
		if cfg.DNSCluster != "" {
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
		urlsmap, _, err = cfg.PeerURLsMapAndToken("proxy")
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

		if gerr != nil {
			plog.Warningf("proxy: %v", gerr)
			return []string{}
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
		Info:    cfg.CorsInfo,
	}

	if cfg.isReadonlyProxy() {
		ph = httpproxy.NewReadonlyHandler(ph)
	}
	// Start a proxy server goroutine for each listen address
	for _, u := range cfg.LCUrls {
		var (
			l      net.Listener
			tlscfg *tls.Config
		)
		if !cfg.ClientTLSInfo.Empty() {
			tlscfg, err = cfg.ClientTLSInfo.ServerConfig()
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

	// capnslog initially SetFormatter(NewDefaultFormatter(os.Stderr))
	// where NewDefaultFormatter returns NewJournaldFormatter when syscall.Getppid() == 1
	// specify 'stdout' or 'stderr' to skip journald logging even when running under systemd
	switch cfg.logOutput {
	case "stdout":
		capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stdout, cfg.Debug))
	case "stderr":
		capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stderr, cfg.Debug))
	case "default":
	default:
		plog.Panicf(`unknown log-output %q (only supports "default", "stdout", "stderr")`, cfg.logOutput)
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
