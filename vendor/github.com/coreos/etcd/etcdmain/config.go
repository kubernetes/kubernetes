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

// Every change should be reflected on help.go as well.

package etcdmain

import (
	"flag"
	"fmt"
	"net/url"
	"os"
	"runtime"
	"strings"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/flags"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/version"
)

const (
	proxyFlagOff      = "off"
	proxyFlagReadonly = "readonly"
	proxyFlagOn       = "on"

	fallbackFlagExit  = "exit"
	fallbackFlagProxy = "proxy"

	clusterStateFlagNew      = "new"
	clusterStateFlagExisting = "existing"

	defaultName                     = "default"
	defaultInitialAdvertisePeerURLs = "http://localhost:2380,http://localhost:7001"

	// maxElectionMs specifies the maximum value of election timeout.
	// More details are listed in ../Documentation/tuning.md#time-parameters.
	maxElectionMs = 50000
)

var (
	ignored = []string{
		"cluster-active-size",
		"cluster-remove-delay",
		"cluster-sync-interval",
		"config",
		"force",
		"max-result-buffer",
		"max-retry-attempts",
		"peer-heartbeat-interval",
		"peer-election-timeout",
		"retry-interval",
		"snapshot",
		"v",
		"vv",
	}

	ErrConflictBootstrapFlags = fmt.Errorf("multiple discovery or bootstrap flags are set. " +
		"Choose one of \"initial-cluster\", \"discovery\" or \"discovery-srv\"")
	errUnsetAdvertiseClientURLsFlag = fmt.Errorf("-advertise-client-urls is required when -listen-client-urls is set explicitly")
)

type config struct {
	*flag.FlagSet

	// member
	corsInfo       *cors.CORSInfo
	dir            string
	walDir         string
	lpurls, lcurls []url.URL
	maxSnapFiles   uint
	maxWalFiles    uint
	name           string
	snapCount      uint64
	// TickMs is the number of milliseconds between heartbeat ticks.
	// TODO: decouple tickMs and heartbeat tick (current heartbeat tick = 1).
	// make ticks a cluster wide configuration.
	TickMs     uint
	ElectionMs uint

	// clustering
	apurls, acurls      []url.URL
	clusterState        *flags.StringsFlag
	dnsCluster          string
	dproxy              string
	durl                string
	fallback            *flags.StringsFlag
	initialCluster      string
	initialClusterToken string
	strictReconfigCheck bool

	// proxy
	proxy                  *flags.StringsFlag
	proxyFailureWaitMs     uint
	proxyRefreshIntervalMs uint
	proxyDialTimeoutMs     uint
	proxyWriteTimeoutMs    uint
	proxyReadTimeoutMs     uint

	// security
	clientTLSInfo, peerTLSInfo transport.TLSInfo

	// logging
	debug        bool
	logPkgLevels string

	// unsafe
	forceNewCluster bool

	printVersion bool

	v3demo                  bool
	gRPCAddr                string
	autoCompactionRetention int

	enablePprof bool

	ignored []string
}

func NewConfig() *config {
	cfg := &config{
		corsInfo: &cors.CORSInfo{},
		clusterState: flags.NewStringsFlag(
			clusterStateFlagNew,
			clusterStateFlagExisting,
		),
		fallback: flags.NewStringsFlag(
			fallbackFlagExit,
			fallbackFlagProxy,
		),
		ignored: ignored,
		proxy: flags.NewStringsFlag(
			proxyFlagOff,
			proxyFlagReadonly,
			proxyFlagOn,
		),
	}

	cfg.FlagSet = flag.NewFlagSet("etcd", flag.ContinueOnError)
	fs := cfg.FlagSet
	fs.Usage = func() {
		fmt.Println(usageline)
	}

	// member
	fs.Var(cfg.corsInfo, "cors", "Comma-separated white list of origins for CORS (cross-origin resource sharing).")
	fs.StringVar(&cfg.dir, "data-dir", "", "Path to the data directory.")
	fs.StringVar(&cfg.walDir, "wal-dir", "", "Path to the dedicated wal directory.")
	fs.Var(flags.NewURLsValue("http://localhost:2380,http://localhost:7001"), "listen-peer-urls", "List of URLs to listen on for peer traffic.")
	fs.Var(flags.NewURLsValue("http://localhost:2379,http://localhost:4001"), "listen-client-urls", "List of URLs to listen on for client traffic.")
	fs.UintVar(&cfg.maxSnapFiles, "max-snapshots", defaultMaxSnapshots, "Maximum number of snapshot files to retain (0 is unlimited).")
	fs.UintVar(&cfg.maxWalFiles, "max-wals", defaultMaxWALs, "Maximum number of wal files to retain (0 is unlimited).")
	fs.StringVar(&cfg.name, "name", defaultName, "Human-readable name for this member.")
	fs.Uint64Var(&cfg.snapCount, "snapshot-count", etcdserver.DefaultSnapCount, "Number of committed transactions to trigger a snapshot to disk.")
	fs.UintVar(&cfg.TickMs, "heartbeat-interval", 100, "Time (in milliseconds) of a heartbeat interval.")
	fs.UintVar(&cfg.ElectionMs, "election-timeout", 1000, "Time (in milliseconds) for an election to timeout.")

	// clustering
	fs.Var(flags.NewURLsValue(defaultInitialAdvertisePeerURLs), "initial-advertise-peer-urls", "List of this member's peer URLs to advertise to the rest of the cluster.")
	fs.Var(flags.NewURLsValue("http://localhost:2379,http://localhost:4001"), "advertise-client-urls", "List of this member's client URLs to advertise to the public.")
	fs.StringVar(&cfg.durl, "discovery", "", "Discovery URL used to bootstrap the cluster.")
	fs.Var(cfg.fallback, "discovery-fallback", fmt.Sprintf("Valid values include %s", strings.Join(cfg.fallback.Values, ", ")))
	if err := cfg.fallback.Set(fallbackFlagProxy); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up discovery-fallback flag: %v", err)
	}
	fs.StringVar(&cfg.dproxy, "discovery-proxy", "", "HTTP proxy to use for traffic to discovery service.")
	fs.StringVar(&cfg.dnsCluster, "discovery-srv", "", "DNS domain used to bootstrap initial cluster.")
	fs.StringVar(&cfg.initialCluster, "initial-cluster", initialClusterFromName(defaultName), "Initial cluster configuration for bootstrapping.")
	fs.StringVar(&cfg.initialClusterToken, "initial-cluster-token", "etcd-cluster", "Initial cluster token for the etcd cluster during bootstrap.")
	fs.Var(cfg.clusterState, "initial-cluster-state", "Initial cluster state ('new' or 'existing').")
	if err := cfg.clusterState.Set(clusterStateFlagNew); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up clusterStateFlag: %v", err)
	}
	fs.BoolVar(&cfg.strictReconfigCheck, "strict-reconfig-check", false, "Reject reconfiguration requests that would cause quorum loss.")

	// proxy
	fs.Var(cfg.proxy, "proxy", fmt.Sprintf("Valid values include %s", strings.Join(cfg.proxy.Values, ", ")))
	if err := cfg.proxy.Set(proxyFlagOff); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up proxyFlag: %v", err)
	}
	fs.UintVar(&cfg.proxyFailureWaitMs, "proxy-failure-wait", 5000, "Time (in milliseconds) an endpoint will be held in a failed state.")
	fs.UintVar(&cfg.proxyRefreshIntervalMs, "proxy-refresh-interval", 30000, "Time (in milliseconds) of the endpoints refresh interval.")
	fs.UintVar(&cfg.proxyDialTimeoutMs, "proxy-dial-timeout", 1000, "Time (in milliseconds) for a dial to timeout.")
	fs.UintVar(&cfg.proxyWriteTimeoutMs, "proxy-write-timeout", 5000, "Time (in milliseconds) for a write to timeout.")
	fs.UintVar(&cfg.proxyReadTimeoutMs, "proxy-read-timeout", 0, "Time (in milliseconds) for a read to timeout.")

	// security
	fs.StringVar(&cfg.clientTLSInfo.CAFile, "ca-file", "", "DEPRECATED: Path to the client server TLS CA file.")
	fs.StringVar(&cfg.clientTLSInfo.CertFile, "cert-file", "", "Path to the client server TLS cert file.")
	fs.StringVar(&cfg.clientTLSInfo.KeyFile, "key-file", "", "Path to the client server TLS key file.")
	fs.BoolVar(&cfg.clientTLSInfo.ClientCertAuth, "client-cert-auth", false, "Enable client cert authentication.")
	fs.StringVar(&cfg.clientTLSInfo.TrustedCAFile, "trusted-ca-file", "", "Path to the client server TLS trusted CA key file.")
	fs.StringVar(&cfg.peerTLSInfo.CAFile, "peer-ca-file", "", "DEPRECATED: Path to the peer server TLS CA file.")
	fs.StringVar(&cfg.peerTLSInfo.CertFile, "peer-cert-file", "", "Path to the peer server TLS cert file.")
	fs.StringVar(&cfg.peerTLSInfo.KeyFile, "peer-key-file", "", "Path to the peer server TLS key file.")
	fs.BoolVar(&cfg.peerTLSInfo.ClientCertAuth, "peer-client-cert-auth", false, "Enable peer client cert authentication.")
	fs.StringVar(&cfg.peerTLSInfo.TrustedCAFile, "peer-trusted-ca-file", "", "Path to the peer server TLS trusted CA file.")

	// logging
	fs.BoolVar(&cfg.debug, "debug", false, "Enable debug-level logging for etcd.")
	fs.StringVar(&cfg.logPkgLevels, "log-package-levels", "", "Specify a particular log level for each etcd package (eg: 'etcdmain=CRITICAL,etcdserver=DEBUG').")

	// unsafe
	fs.BoolVar(&cfg.forceNewCluster, "force-new-cluster", false, "Force to create a new one member cluster.")

	// version
	fs.BoolVar(&cfg.printVersion, "version", false, "Print the version and exit.")

	// demo flag
	fs.BoolVar(&cfg.v3demo, "experimental-v3demo", false, "Enable experimental v3 demo API.")
	fs.StringVar(&cfg.gRPCAddr, "experimental-gRPC-addr", "127.0.0.1:2378", "gRPC address for experimental v3 demo API.")
	fs.IntVar(&cfg.autoCompactionRetention, "experimental-auto-compaction-retention", 0, "Auto compaction retention in hour. 0 means disable auto compaction.")

	// backwards-compatibility with v0.4.6
	fs.Var(&flags.IPAddressPort{}, "addr", "DEPRECATED: Use -advertise-client-urls instead.")
	fs.Var(&flags.IPAddressPort{}, "bind-addr", "DEPRECATED: Use -listen-client-urls instead.")
	fs.Var(&flags.IPAddressPort{}, "peer-addr", "DEPRECATED: Use -initial-advertise-peer-urls instead.")
	fs.Var(&flags.IPAddressPort{}, "peer-bind-addr", "DEPRECATED: Use -listen-peer-urls instead.")
	fs.Var(&flags.DeprecatedFlag{Name: "peers"}, "peers", "DEPRECATED: Use -initial-cluster instead.")
	fs.Var(&flags.DeprecatedFlag{Name: "peers-file"}, "peers-file", "DEPRECATED: Use -initial-cluster instead.")

	// pprof profiler via HTTP
	fs.BoolVar(&cfg.enablePprof, "enable-pprof", false, "Enable runtime profiling data via HTTP server. Address is at client URL + \"/debug/pprof\"")

	// ignored
	for _, f := range cfg.ignored {
		fs.Var(&flags.IgnoredFlag{Name: f}, f, "")
	}
	return cfg
}

func (cfg *config) Parse(arguments []string) error {
	perr := cfg.FlagSet.Parse(arguments)
	switch perr {
	case nil:
	case flag.ErrHelp:
		fmt.Println(flagsline)
		os.Exit(0)
	default:
		os.Exit(2)
	}
	if len(cfg.FlagSet.Args()) != 0 {
		return fmt.Errorf("'%s' is not a valid flag", cfg.FlagSet.Arg(0))
	}

	if cfg.printVersion {
		fmt.Printf("etcd Version: %s\n", version.Version)
		fmt.Printf("Git SHA: %s\n", version.GitSHA)
		fmt.Printf("Go Version: %s\n", runtime.Version())
		fmt.Printf("Go OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)
		os.Exit(0)
	}

	err := flags.SetFlagsFromEnv(cfg.FlagSet)
	if err != nil {
		plog.Fatalf("%v", err)
	}

	set := make(map[string]bool)
	cfg.FlagSet.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})
	nSet := 0
	for _, v := range []bool{set["discovery"], set["initial-cluster"], set["discovery-srv"]} {
		if v {
			nSet += 1
		}
	}
	if nSet > 1 {
		return ErrConflictBootstrapFlags
	}

	flags.SetBindAddrFromAddr(cfg.FlagSet, "peer-bind-addr", "peer-addr")
	flags.SetBindAddrFromAddr(cfg.FlagSet, "bind-addr", "addr")

	cfg.lpurls, err = flags.URLsFromFlags(cfg.FlagSet, "listen-peer-urls", "peer-bind-addr", cfg.peerTLSInfo)
	if err != nil {
		return err
	}
	cfg.apurls, err = flags.URLsFromFlags(cfg.FlagSet, "initial-advertise-peer-urls", "peer-addr", cfg.peerTLSInfo)
	if err != nil {
		return err
	}
	cfg.lcurls, err = flags.URLsFromFlags(cfg.FlagSet, "listen-client-urls", "bind-addr", cfg.clientTLSInfo)
	if err != nil {
		return err
	}
	cfg.acurls, err = flags.URLsFromFlags(cfg.FlagSet, "advertise-client-urls", "addr", cfg.clientTLSInfo)
	if err != nil {
		return err
	}

	// when etcd runs in member mode user needs to set -advertise-client-urls if -listen-client-urls is set.
	// TODO(yichengq): check this for joining through discovery service case
	mayFallbackToProxy := flags.IsSet(cfg.FlagSet, "discovery") && cfg.fallback.String() == fallbackFlagProxy
	mayBeProxy := cfg.proxy.String() != proxyFlagOff || mayFallbackToProxy
	if !mayBeProxy {
		if flags.IsSet(cfg.FlagSet, "listen-client-urls") && !flags.IsSet(cfg.FlagSet, "advertise-client-urls") {
			return errUnsetAdvertiseClientURLsFlag
		}
	}

	if 5*cfg.TickMs > cfg.ElectionMs {
		return fmt.Errorf("-election-timeout[%vms] should be at least as 5 times as -heartbeat-interval[%vms]", cfg.ElectionMs, cfg.TickMs)
	}
	if cfg.ElectionMs > maxElectionMs {
		return fmt.Errorf("-election-timeout[%vms] is too long, and should be set less than %vms", cfg.ElectionMs, maxElectionMs)
	}

	return nil
}

func initialClusterFromName(name string) string {
	n := name
	if name == "" {
		n = defaultName
	}
	return fmt.Sprintf("%s=http://localhost:2380,%s=http://localhost:7001", n, n)
}

func (cfg config) isNewCluster() bool          { return cfg.clusterState.String() == clusterStateFlagNew }
func (cfg config) isProxy() bool               { return cfg.proxy.String() != proxyFlagOff }
func (cfg config) isReadonlyProxy() bool       { return cfg.proxy.String() == proxyFlagReadonly }
func (cfg config) shouldFallbackToProxy() bool { return cfg.fallback.String() == fallbackFlagProxy }

func (cfg config) electionTicks() int { return int(cfg.ElectionMs / cfg.TickMs) }
