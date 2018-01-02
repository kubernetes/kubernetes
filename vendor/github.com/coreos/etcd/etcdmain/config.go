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

// Every change should be reflected on help.go as well.

package etcdmain

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"strings"

	"github.com/coreos/etcd/embed"
	"github.com/coreos/etcd/pkg/flags"
	"github.com/coreos/etcd/version"
	"github.com/ghodss/yaml"
)

var (
	proxyFlagOff      = "off"
	proxyFlagReadonly = "readonly"
	proxyFlagOn       = "on"

	fallbackFlagExit  = "exit"
	fallbackFlagProxy = "proxy"

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
)

type configProxy struct {
	ProxyFailureWaitMs     uint `json:"proxy-failure-wait"`
	ProxyRefreshIntervalMs uint `json:"proxy-refresh-interval"`
	ProxyDialTimeoutMs     uint `json:"proxy-dial-timeout"`
	ProxyWriteTimeoutMs    uint `json:"proxy-write-timeout"`
	ProxyReadTimeoutMs     uint `json:"proxy-read-timeout"`
	Fallback               string
	Proxy                  string
	ProxyJSON              string `json:"proxy"`
	FallbackJSON           string `json:"discovery-fallback"`
}

// config holds the config for a command line invocation of etcd
type config struct {
	embed.Config
	configProxy
	configFlags
	configFile   string
	printVersion bool
	ignored      []string
	logOutput    string
}

// configFlags has the set of flags used for command line parsing a Config
type configFlags struct {
	*flag.FlagSet
	clusterState *flags.StringsFlag
	fallback     *flags.StringsFlag
	proxy        *flags.StringsFlag
}

func newConfig() *config {
	cfg := &config{
		Config: *embed.NewConfig(),
		configProxy: configProxy{
			Proxy:                  proxyFlagOff,
			ProxyFailureWaitMs:     5000,
			ProxyRefreshIntervalMs: 30000,
			ProxyDialTimeoutMs:     1000,
			ProxyWriteTimeoutMs:    5000,
		},
		ignored: ignored,
	}
	cfg.configFlags = configFlags{
		FlagSet: flag.NewFlagSet("etcd", flag.ContinueOnError),
		clusterState: flags.NewStringsFlag(
			embed.ClusterStateFlagNew,
			embed.ClusterStateFlagExisting,
		),
		fallback: flags.NewStringsFlag(
			fallbackFlagExit,
			fallbackFlagProxy,
		),
		proxy: flags.NewStringsFlag(
			proxyFlagOff,
			proxyFlagReadonly,
			proxyFlagOn,
		),
	}

	fs := cfg.FlagSet
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, usageline)
	}

	fs.StringVar(&cfg.configFile, "config-file", "", "Path to the server configuration file")

	// member
	fs.Var(cfg.CorsInfo, "cors", "Comma-separated white list of origins for CORS (cross-origin resource sharing).")
	fs.StringVar(&cfg.Dir, "data-dir", cfg.Dir, "Path to the data directory.")
	fs.StringVar(&cfg.WalDir, "wal-dir", cfg.WalDir, "Path to the dedicated wal directory.")
	fs.Var(flags.NewURLsValue(embed.DefaultListenPeerURLs), "listen-peer-urls", "List of URLs to listen on for peer traffic.")
	fs.Var(flags.NewURLsValue(embed.DefaultListenClientURLs), "listen-client-urls", "List of URLs to listen on for client traffic.")
	fs.UintVar(&cfg.MaxSnapFiles, "max-snapshots", cfg.MaxSnapFiles, "Maximum number of snapshot files to retain (0 is unlimited).")
	fs.UintVar(&cfg.MaxWalFiles, "max-wals", cfg.MaxWalFiles, "Maximum number of wal files to retain (0 is unlimited).")
	fs.StringVar(&cfg.Name, "name", cfg.Name, "Human-readable name for this member.")
	fs.Uint64Var(&cfg.SnapCount, "snapshot-count", cfg.SnapCount, "Number of committed transactions to trigger a snapshot to disk.")
	fs.UintVar(&cfg.TickMs, "heartbeat-interval", cfg.TickMs, "Time (in milliseconds) of a heartbeat interval.")
	fs.UintVar(&cfg.ElectionMs, "election-timeout", cfg.ElectionMs, "Time (in milliseconds) for an election to timeout.")
	fs.Int64Var(&cfg.QuotaBackendBytes, "quota-backend-bytes", cfg.QuotaBackendBytes, "Raise alarms when backend size exceeds the given quota. 0 means use the default quota.")

	// clustering
	fs.Var(flags.NewURLsValue(embed.DefaultInitialAdvertisePeerURLs), "initial-advertise-peer-urls", "List of this member's peer URLs to advertise to the rest of the cluster.")
	fs.Var(flags.NewURLsValue(embed.DefaultAdvertiseClientURLs), "advertise-client-urls", "List of this member's client URLs to advertise to the public.")
	fs.StringVar(&cfg.Durl, "discovery", cfg.Durl, "Discovery URL used to bootstrap the cluster.")
	fs.Var(cfg.fallback, "discovery-fallback", fmt.Sprintf("Valid values include %s", strings.Join(cfg.fallback.Values, ", ")))
	if err := cfg.fallback.Set(fallbackFlagProxy); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up discovery-fallback flag: %v", err)
	}
	fs.StringVar(&cfg.Dproxy, "discovery-proxy", cfg.Dproxy, "HTTP proxy to use for traffic to discovery service.")
	fs.StringVar(&cfg.DNSCluster, "discovery-srv", cfg.DNSCluster, "DNS domain used to bootstrap initial cluster.")
	fs.StringVar(&cfg.InitialCluster, "initial-cluster", cfg.InitialCluster, "Initial cluster configuration for bootstrapping.")
	fs.StringVar(&cfg.InitialClusterToken, "initial-cluster-token", cfg.InitialClusterToken, "Initial cluster token for the etcd cluster during bootstrap.")
	fs.Var(cfg.clusterState, "initial-cluster-state", "Initial cluster state ('new' or 'existing').")
	if err := cfg.clusterState.Set(embed.ClusterStateFlagNew); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up clusterStateFlag: %v", err)
	}
	fs.BoolVar(&cfg.StrictReconfigCheck, "strict-reconfig-check", cfg.StrictReconfigCheck, "Reject reconfiguration requests that would cause quorum loss.")

	// proxy
	fs.Var(cfg.proxy, "proxy", fmt.Sprintf("Valid values include %s", strings.Join(cfg.proxy.Values, ", ")))
	if err := cfg.proxy.Set(proxyFlagOff); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up proxyFlag: %v", err)
	}
	fs.UintVar(&cfg.ProxyFailureWaitMs, "proxy-failure-wait", cfg.ProxyFailureWaitMs, "Time (in milliseconds) an endpoint will be held in a failed state.")
	fs.UintVar(&cfg.ProxyRefreshIntervalMs, "proxy-refresh-interval", cfg.ProxyRefreshIntervalMs, "Time (in milliseconds) of the endpoints refresh interval.")
	fs.UintVar(&cfg.ProxyDialTimeoutMs, "proxy-dial-timeout", cfg.ProxyDialTimeoutMs, "Time (in milliseconds) for a dial to timeout.")
	fs.UintVar(&cfg.ProxyWriteTimeoutMs, "proxy-write-timeout", cfg.ProxyWriteTimeoutMs, "Time (in milliseconds) for a write to timeout.")
	fs.UintVar(&cfg.ProxyReadTimeoutMs, "proxy-read-timeout", cfg.ProxyReadTimeoutMs, "Time (in milliseconds) for a read to timeout.")

	// security
	fs.StringVar(&cfg.ClientTLSInfo.CAFile, "ca-file", "", "DEPRECATED: Path to the client server TLS CA file.")
	fs.StringVar(&cfg.ClientTLSInfo.CertFile, "cert-file", "", "Path to the client server TLS cert file.")
	fs.StringVar(&cfg.ClientTLSInfo.KeyFile, "key-file", "", "Path to the client server TLS key file.")
	fs.BoolVar(&cfg.ClientTLSInfo.ClientCertAuth, "client-cert-auth", false, "Enable client cert authentication.")
	fs.StringVar(&cfg.ClientTLSInfo.TrustedCAFile, "trusted-ca-file", "", "Path to the client server TLS trusted CA key file.")
	fs.BoolVar(&cfg.ClientAutoTLS, "auto-tls", false, "Client TLS using generated certificates")
	fs.StringVar(&cfg.PeerTLSInfo.CAFile, "peer-ca-file", "", "DEPRECATED: Path to the peer server TLS CA file.")
	fs.StringVar(&cfg.PeerTLSInfo.CertFile, "peer-cert-file", "", "Path to the peer server TLS cert file.")
	fs.StringVar(&cfg.PeerTLSInfo.KeyFile, "peer-key-file", "", "Path to the peer server TLS key file.")
	fs.BoolVar(&cfg.PeerTLSInfo.ClientCertAuth, "peer-client-cert-auth", false, "Enable peer client cert authentication.")
	fs.StringVar(&cfg.PeerTLSInfo.TrustedCAFile, "peer-trusted-ca-file", "", "Path to the peer server TLS trusted CA file.")
	fs.BoolVar(&cfg.PeerAutoTLS, "peer-auto-tls", false, "Peer TLS using generated certificates")

	// logging
	fs.BoolVar(&cfg.Debug, "debug", false, "Enable debug-level logging for etcd.")
	fs.StringVar(&cfg.LogPkgLevels, "log-package-levels", "", "Specify a particular log level for each etcd package (eg: 'etcdmain=CRITICAL,etcdserver=DEBUG').")
	fs.StringVar(&cfg.logOutput, "log-output", "default", "Specify 'stdout' or 'stderr' to skip journald logging even when running under systemd.")

	// unsafe
	fs.BoolVar(&cfg.ForceNewCluster, "force-new-cluster", false, "Force to create a new one member cluster.")

	// version
	fs.BoolVar(&cfg.printVersion, "version", false, "Print the version and exit.")

	fs.IntVar(&cfg.AutoCompactionRetention, "auto-compaction-retention", 0, "Auto compaction retention for mvcc key value store in hour. 0 means disable auto compaction.")

	// pprof profiler via HTTP
	fs.BoolVar(&cfg.EnablePprof, "enable-pprof", false, "Enable runtime profiling data via HTTP server. Address is at client URL + \"/debug/pprof/\"")

	// additional metrics
	fs.StringVar(&cfg.Metrics, "metrics", cfg.Metrics, "Set level of detail for exported metrics, specify 'extensive' to include histogram metrics")

	// ignored
	for _, f := range cfg.ignored {
		fs.Var(&flags.IgnoredFlag{Name: f}, f, "")
	}
	return cfg
}

func (cfg *config) parse(arguments []string) error {
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

	var err error
	if cfg.configFile != "" {
		plog.Infof("Loading server configuration from %q", cfg.configFile)
		err = cfg.configFromFile(cfg.configFile)
	} else {
		err = cfg.configFromCmdLine()
	}
	return err
}

func (cfg *config) configFromCmdLine() error {
	err := flags.SetFlagsFromEnv("ETCD", cfg.FlagSet)
	if err != nil {
		plog.Fatalf("%v", err)
	}

	cfg.LPUrls = flags.URLsFromFlag(cfg.FlagSet, "listen-peer-urls")
	cfg.APUrls = flags.URLsFromFlag(cfg.FlagSet, "initial-advertise-peer-urls")
	cfg.LCUrls = flags.URLsFromFlag(cfg.FlagSet, "listen-client-urls")
	cfg.ACUrls = flags.URLsFromFlag(cfg.FlagSet, "advertise-client-urls")
	cfg.ClusterState = cfg.clusterState.String()
	cfg.Fallback = cfg.fallback.String()
	cfg.Proxy = cfg.proxy.String()

	// disable default advertise-client-urls if lcurls is set
	missingAC := flags.IsSet(cfg.FlagSet, "listen-client-urls") && !flags.IsSet(cfg.FlagSet, "advertise-client-urls")
	if !cfg.mayBeProxy() && missingAC {
		cfg.ACUrls = nil
	}

	// disable default initial-cluster if discovery is set
	if (cfg.Durl != "" || cfg.DNSCluster != "") && !flags.IsSet(cfg.FlagSet, "initial-cluster") {
		cfg.InitialCluster = ""
	}

	return cfg.validate()
}

func (cfg *config) configFromFile(path string) error {
	eCfg, err := embed.ConfigFromFile(path)
	if err != nil {
		return err
	}
	cfg.Config = *eCfg

	// load extra config information
	b, rerr := ioutil.ReadFile(path)
	if rerr != nil {
		return rerr
	}
	if yerr := yaml.Unmarshal(b, &cfg.configProxy); yerr != nil {
		return yerr
	}
	if cfg.FallbackJSON != "" {
		if err := cfg.fallback.Set(cfg.FallbackJSON); err != nil {
			plog.Panicf("unexpected error setting up discovery-fallback flag: %v", err)
		}
		cfg.Fallback = cfg.fallback.String()
	}
	if cfg.ProxyJSON != "" {
		if err := cfg.proxy.Set(cfg.ProxyJSON); err != nil {
			plog.Panicf("unexpected error setting up proxyFlag: %v", err)
		}
		cfg.Proxy = cfg.proxy.String()
	}
	return nil
}

func (cfg *config) mayBeProxy() bool {
	mayFallbackToProxy := cfg.Durl != "" && cfg.Fallback == fallbackFlagProxy
	return cfg.Proxy != proxyFlagOff || mayFallbackToProxy
}

func (cfg *config) validate() error {
	err := cfg.Config.Validate()
	// TODO(yichengq): check this for joining through discovery service case
	if err == embed.ErrUnsetAdvertiseClientURLsFlag && cfg.mayBeProxy() {
		return nil
	}
	return err
}

func (cfg config) isProxy() bool               { return cfg.proxy.String() != proxyFlagOff }
func (cfg config) isReadonlyProxy() bool       { return cfg.proxy.String() == proxyFlagReadonly }
func (cfg config) shouldFallbackToProxy() bool { return cfg.fallback.String() == fallbackFlagProxy }
