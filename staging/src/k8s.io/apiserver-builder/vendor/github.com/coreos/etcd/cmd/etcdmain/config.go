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
	"net"
	"net/url"
	"os"
	"runtime"
	"strings"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/flags"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/version"
	"github.com/ghodss/yaml"
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
	defaultInitialAdvertisePeerURLs = "http://localhost:2380"
	defaultAdvertiseClientURLs      = "http://localhost:2379"
	defaultListenPeerURLs           = "http://localhost:2380"
	defaultListenClientURLs         = "http://localhost:2379"

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
	errUnsetAdvertiseClientURLsFlag = fmt.Errorf("--advertise-client-urls is required when --listen-client-urls is set explicitly")
)

type config struct {
	*flag.FlagSet

	// member
	corsInfo       *cors.CORSInfo
	lpurls, lcurls []url.URL
	Dir            string `json:"data-dir"`
	WalDir         string `json:"wal-dir"`
	MaxSnapFiles   uint   `json:"max-snapshots"`
	MaxWalFiles    uint   `json:"max-wals"`
	Name           string `json:"name"`
	SnapCount      uint64 `json:"snapshot-count"`
	LPUrlsCfgFile  string `json:"listen-peer-urls"`
	LCUrlsCfgFile  string `json:"listen-client-urls"`
	CorsCfgFile    string `json:"cors"`

	// TickMs is the number of milliseconds between heartbeat ticks.
	// TODO: decouple tickMs and heartbeat tick (current heartbeat tick = 1).
	// make ticks a cluster wide configuration.
	TickMs            uint  `json:"heartbeat-interval"`
	ElectionMs        uint  `json:"election-timeout"`
	QuotaBackendBytes int64 `json:"quota-backend-bytes"`

	// clustering
	apurls, acurls      []url.URL
	clusterState        *flags.StringsFlag
	DnsCluster          string `json:"discovery-srv"`
	Dproxy              string `json:"discovery-proxy"`
	Durl                string `json:"discovery"`
	fallback            *flags.StringsFlag
	InitialCluster      string `json:"initial-cluster"`
	InitialClusterToken string `json:"initial-cluster-token"`
	StrictReconfigCheck bool   `json:"strict-reconfig-check"`
	ApurlsCfgFile       string `json:"initial-advertise-peer-urls"`
	AcurlsCfgFile       string `json:"advertise-client-urls"`
	ClusterStateCfgFile string `json:"initial-cluster-state"`
	FallbackCfgFile     string `json:"discovery-fallback"`

	// proxy
	proxy                  *flags.StringsFlag
	ProxyFailureWaitMs     uint   `json:"proxy-failure-wait"`
	ProxyRefreshIntervalMs uint   `json:"proxy-refresh-interval"`
	ProxyDialTimeoutMs     uint   `json:"proxy-dial-timeout"`
	ProxyWriteTimeoutMs    uint   `json:"proxy-write-timeout"`
	ProxyReadTimeoutMs     uint   `json:"proxy-read-timeout"`
	ProxyCfgFile           string `json:"proxy"`

	// security
	clientTLSInfo, peerTLSInfo transport.TLSInfo
	ClientAutoTLS              bool
	PeerAutoTLS                bool
	ClientSecurityCfgFile      securityConfig `json:"client-transport-security"`
	PeerSecurityCfgFile        securityConfig `json:"peer-transport-security"`

	// Debug logging
	Debug        bool   `json:"debug"`
	LogPkgLevels string `json:"log-package-levels"`

	// ForceNewCluster is unsafe
	ForceNewCluster bool `json:"force-new-cluster"`

	printVersion bool

	autoCompactionRetention int

	enablePprof bool

	configFile string

	ignored []string
}

type securityConfig struct {
	CAFile        string `json:"ca-file"`
	CertFile      string `json:"cert-file"`
	KeyFile       string `json:"key-file"`
	CertAuth      bool   `json:"client-cert-auth"`
	TrustedCAFile string `json:"trusted-ca-file"`
	AutoTLS       bool   `json:"auto-tls"`
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

	fs.StringVar(&cfg.configFile, "config-file", "", "Path to the server configuration file")

	// member
	fs.Var(cfg.corsInfo, "cors", "Comma-separated white list of origins for CORS (cross-origin resource sharing).")
	fs.StringVar(&cfg.Dir, "data-dir", "", "Path to the data directory.")
	fs.StringVar(&cfg.WalDir, "wal-dir", "", "Path to the dedicated wal directory.")
	fs.Var(flags.NewURLsValue(defaultListenPeerURLs), "listen-peer-urls", "List of URLs to listen on for peer traffic.")
	fs.Var(flags.NewURLsValue(defaultListenClientURLs), "listen-client-urls", "List of URLs to listen on for client traffic.")
	fs.UintVar(&cfg.MaxSnapFiles, "max-snapshots", defaultMaxSnapshots, "Maximum number of snapshot files to retain (0 is unlimited).")
	fs.UintVar(&cfg.MaxWalFiles, "max-wals", defaultMaxWALs, "Maximum number of wal files to retain (0 is unlimited).")
	fs.StringVar(&cfg.Name, "name", defaultName, "Human-readable name for this member.")
	fs.Uint64Var(&cfg.SnapCount, "snapshot-count", etcdserver.DefaultSnapCount, "Number of committed transactions to trigger a snapshot to disk.")
	fs.UintVar(&cfg.TickMs, "heartbeat-interval", 100, "Time (in milliseconds) of a heartbeat interval.")
	fs.UintVar(&cfg.ElectionMs, "election-timeout", 1000, "Time (in milliseconds) for an election to timeout.")
	fs.Int64Var(&cfg.QuotaBackendBytes, "quota-backend-bytes", 0, "Raise alarms when backend size exceeds the given quota. 0 means use the default quota.")

	// clustering
	fs.Var(flags.NewURLsValue(defaultInitialAdvertisePeerURLs), "initial-advertise-peer-urls", "List of this member's peer URLs to advertise to the rest of the cluster.")
	fs.Var(flags.NewURLsValue(defaultAdvertiseClientURLs), "advertise-client-urls", "List of this member's client URLs to advertise to the public.")
	fs.StringVar(&cfg.Durl, "discovery", "", "Discovery URL used to bootstrap the cluster.")
	fs.Var(cfg.fallback, "discovery-fallback", fmt.Sprintf("Valid values include %s", strings.Join(cfg.fallback.Values, ", ")))
	if err := cfg.fallback.Set(fallbackFlagProxy); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up discovery-fallback flag: %v", err)
	}
	fs.StringVar(&cfg.Dproxy, "discovery-proxy", "", "HTTP proxy to use for traffic to discovery service.")
	fs.StringVar(&cfg.DnsCluster, "discovery-srv", "", "DNS domain used to bootstrap initial cluster.")
	fs.StringVar(&cfg.InitialCluster, "initial-cluster", initialClusterFromName(defaultName), "Initial cluster configuration for bootstrapping.")
	fs.StringVar(&cfg.InitialClusterToken, "initial-cluster-token", "etcd-cluster", "Initial cluster token for the etcd cluster during bootstrap.")
	fs.Var(cfg.clusterState, "initial-cluster-state", "Initial cluster state ('new' or 'existing').")
	if err := cfg.clusterState.Set(clusterStateFlagNew); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up clusterStateFlag: %v", err)
	}
	fs.BoolVar(&cfg.StrictReconfigCheck, "strict-reconfig-check", false, "Reject reconfiguration requests that would cause quorum loss.")

	// proxy
	fs.Var(cfg.proxy, "proxy", fmt.Sprintf("Valid values include %s", strings.Join(cfg.proxy.Values, ", ")))
	if err := cfg.proxy.Set(proxyFlagOff); err != nil {
		// Should never happen.
		plog.Panicf("unexpected error setting up proxyFlag: %v", err)
	}
	fs.UintVar(&cfg.ProxyFailureWaitMs, "proxy-failure-wait", 5000, "Time (in milliseconds) an endpoint will be held in a failed state.")
	fs.UintVar(&cfg.ProxyRefreshIntervalMs, "proxy-refresh-interval", 30000, "Time (in milliseconds) of the endpoints refresh interval.")
	fs.UintVar(&cfg.ProxyDialTimeoutMs, "proxy-dial-timeout", 1000, "Time (in milliseconds) for a dial to timeout.")
	fs.UintVar(&cfg.ProxyWriteTimeoutMs, "proxy-write-timeout", 5000, "Time (in milliseconds) for a write to timeout.")
	fs.UintVar(&cfg.ProxyReadTimeoutMs, "proxy-read-timeout", 0, "Time (in milliseconds) for a read to timeout.")

	// security
	fs.StringVar(&cfg.clientTLSInfo.CAFile, "ca-file", "", "DEPRECATED: Path to the client server TLS CA file.")
	fs.StringVar(&cfg.clientTLSInfo.CertFile, "cert-file", "", "Path to the client server TLS cert file.")
	fs.StringVar(&cfg.clientTLSInfo.KeyFile, "key-file", "", "Path to the client server TLS key file.")
	fs.BoolVar(&cfg.clientTLSInfo.ClientCertAuth, "client-cert-auth", false, "Enable client cert authentication.")
	fs.StringVar(&cfg.clientTLSInfo.TrustedCAFile, "trusted-ca-file", "", "Path to the client server TLS trusted CA key file.")
	fs.BoolVar(&cfg.ClientAutoTLS, "auto-tls", false, "Client TLS using generated certificates")
	fs.StringVar(&cfg.peerTLSInfo.CAFile, "peer-ca-file", "", "DEPRECATED: Path to the peer server TLS CA file.")
	fs.StringVar(&cfg.peerTLSInfo.CertFile, "peer-cert-file", "", "Path to the peer server TLS cert file.")
	fs.StringVar(&cfg.peerTLSInfo.KeyFile, "peer-key-file", "", "Path to the peer server TLS key file.")
	fs.BoolVar(&cfg.peerTLSInfo.ClientCertAuth, "peer-client-cert-auth", false, "Enable peer client cert authentication.")
	fs.StringVar(&cfg.peerTLSInfo.TrustedCAFile, "peer-trusted-ca-file", "", "Path to the peer server TLS trusted CA file.")
	fs.BoolVar(&cfg.PeerAutoTLS, "peer-auto-tls", false, "Peer TLS using generated certificates")

	// logging
	fs.BoolVar(&cfg.Debug, "debug", false, "Enable debug-level logging for etcd.")
	fs.StringVar(&cfg.LogPkgLevels, "log-package-levels", "", "Specify a particular log level for each etcd package (eg: 'etcdmain=CRITICAL,etcdserver=DEBUG').")

	// unsafe
	fs.BoolVar(&cfg.ForceNewCluster, "force-new-cluster", false, "Force to create a new one member cluster.")

	// version
	fs.BoolVar(&cfg.printVersion, "version", false, "Print the version and exit.")

	fs.IntVar(&cfg.autoCompactionRetention, "auto-compaction-retention", 0, "Auto compaction retention for mvcc key value store in hour. 0 means disable auto compaction.")

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

	var err error
	if cfg.configFile != "" {
		plog.Infof("Loading server configuration from %q", cfg.configFile)
		err = cfg.configFromFile()
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

	cfg.lpurls = flags.URLsFromFlag(cfg.FlagSet, "listen-peer-urls")
	cfg.apurls = flags.URLsFromFlag(cfg.FlagSet, "initial-advertise-peer-urls")
	cfg.lcurls = flags.URLsFromFlag(cfg.FlagSet, "listen-client-urls")
	cfg.acurls = flags.URLsFromFlag(cfg.FlagSet, "advertise-client-urls")

	return cfg.validateConfig(func(field string) bool {
		return flags.IsSet(cfg.FlagSet, field)
	})
}

func (cfg *config) configFromFile() error {
	b, err := ioutil.ReadFile(cfg.configFile)
	if err != nil {
		return err
	}

	err = yaml.Unmarshal(b, cfg)
	if err != nil {
		return err
	}

	if cfg.LPUrlsCfgFile != "" {
		u, err := types.NewURLs(strings.Split(cfg.LPUrlsCfgFile, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up listen-peer-urls: %v", err)
		}
		cfg.lpurls = []url.URL(u)
	}

	if cfg.LCUrlsCfgFile != "" {
		u, err := types.NewURLs(strings.Split(cfg.LCUrlsCfgFile, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up listen-client-urls: %v", err)
		}
		cfg.lcurls = []url.URL(u)
	}

	if cfg.CorsCfgFile != "" {
		if err := cfg.corsInfo.Set(cfg.CorsCfgFile); err != nil {
			plog.Panicf("unexpected error setting up cors: %v", err)
		}
	}

	if cfg.ApurlsCfgFile != "" {
		u, err := types.NewURLs(strings.Split(cfg.ApurlsCfgFile, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up initial-advertise-peer-urls: %v", err)
		}
		cfg.apurls = []url.URL(u)
	}

	if cfg.AcurlsCfgFile != "" {
		u, err := types.NewURLs(strings.Split(cfg.AcurlsCfgFile, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up advertise-peer-urls: %v", err)
		}
		cfg.acurls = []url.URL(u)
	}

	if cfg.ClusterStateCfgFile != "" {
		if err := cfg.clusterState.Set(cfg.ClusterStateCfgFile); err != nil {
			plog.Panicf("unexpected error setting up clusterStateFlag: %v", err)
		}
	}

	if cfg.FallbackCfgFile != "" {
		if err := cfg.fallback.Set(cfg.FallbackCfgFile); err != nil {
			plog.Panicf("unexpected error setting up discovery-fallback flag: %v", err)
		}
	}

	if cfg.ProxyCfgFile != "" {
		if err := cfg.proxy.Set(cfg.ProxyCfgFile); err != nil {
			plog.Panicf("unexpected error setting up proxyFlag: %v", err)
		}
	}

	copySecurityDetails := func(tls *transport.TLSInfo, ysc *securityConfig) {
		tls.CAFile = ysc.CAFile
		tls.CertFile = ysc.CertFile
		tls.KeyFile = ysc.KeyFile
		tls.ClientCertAuth = ysc.CertAuth
		tls.TrustedCAFile = ysc.TrustedCAFile
	}
	copySecurityDetails(&cfg.clientTLSInfo, &cfg.ClientSecurityCfgFile)
	copySecurityDetails(&cfg.peerTLSInfo, &cfg.PeerSecurityCfgFile)
	cfg.ClientAutoTLS = cfg.ClientSecurityCfgFile.AutoTLS
	cfg.PeerAutoTLS = cfg.PeerSecurityCfgFile.AutoTLS

	fieldsToBeChecked := map[string]bool{
		"discovery":             (cfg.Durl != ""),
		"listen-client-urls":    (cfg.LCUrlsCfgFile != ""),
		"advertise-client-urls": (cfg.AcurlsCfgFile != ""),
		"initial-cluster":       (cfg.InitialCluster != ""),
		"discovery-srv":         (cfg.DnsCluster != ""),
	}

	return cfg.validateConfig(func(field string) bool {
		return fieldsToBeChecked[field]
	})
}

func (cfg *config) validateConfig(isSet func(field string) bool) error {
	if err := checkBindURLs(cfg.lpurls); err != nil {
		return err
	}
	if err := checkBindURLs(cfg.lcurls); err != nil {
		return err
	}

	// when etcd runs in member mode user needs to set --advertise-client-urls if --listen-client-urls is set.
	// TODO(yichengq): check this for joining through discovery service case
	mayFallbackToProxy := isSet("discovery") && cfg.fallback.String() == fallbackFlagProxy
	mayBeProxy := cfg.proxy.String() != proxyFlagOff || mayFallbackToProxy
	if !mayBeProxy {
		if isSet("listen-client-urls") && !isSet("advertise-client-urls") {
			return errUnsetAdvertiseClientURLsFlag
		}
	}

	// Check if conflicting flags are passed.
	nSet := 0
	for _, v := range []bool{isSet("discovery"), isSet("initial-cluster"), isSet("discovery-srv")} {
		if v {
			nSet += 1
		}
	}

	if nSet > 1 {
		return ErrConflictBootstrapFlags
	}

	if 5*cfg.TickMs > cfg.ElectionMs {
		return fmt.Errorf("--election-timeout[%vms] should be at least as 5 times as --heartbeat-interval[%vms]", cfg.ElectionMs, cfg.TickMs)
	}
	if cfg.ElectionMs > maxElectionMs {
		return fmt.Errorf("--election-timeout[%vms] is too long, and should be set less than %vms", cfg.ElectionMs, maxElectionMs)
	}

	return nil
}

func initialClusterFromName(name string) string {
	n := name
	if name == "" {
		n = defaultName
	}
	return fmt.Sprintf("%s=http://localhost:2380", n)
}

func (cfg config) isNewCluster() bool          { return cfg.clusterState.String() == clusterStateFlagNew }
func (cfg config) isProxy() bool               { return cfg.proxy.String() != proxyFlagOff }
func (cfg config) isReadonlyProxy() bool       { return cfg.proxy.String() == proxyFlagReadonly }
func (cfg config) shouldFallbackToProxy() bool { return cfg.fallback.String() == fallbackFlagProxy }

func (cfg config) electionTicks() int { return int(cfg.ElectionMs / cfg.TickMs) }

// checkBindURLs returns an error if any URL uses a domain name.
// TODO: return error in 3.2.0
func checkBindURLs(urls []url.URL) error {
	for _, url := range urls {
		if url.Scheme == "unix" || url.Scheme == "unixs" {
			continue
		}
		host, _, err := net.SplitHostPort(url.Host)
		if err != nil {
			return err
		}
		if host == "localhost" {
			// special case for local address
			// TODO: support /etc/hosts ?
			continue
		}
		if net.ParseIP(host) == nil {
			err := fmt.Errorf("expected IP in URL for binding (%s)", url.String())
			plog.Warning(err)
		}
	}
	return nil
}
