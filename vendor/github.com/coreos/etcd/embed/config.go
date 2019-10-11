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
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coreos/etcd/compactor"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/netutil"
	"github.com/coreos/etcd/pkg/srv"
	"github.com/coreos/etcd/pkg/tlsutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"

	"github.com/coreos/pkg/capnslog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"sigs.k8s.io/yaml"
)

const (
	ClusterStateFlagNew      = "new"
	ClusterStateFlagExisting = "existing"

	DefaultName                  = "default"
	DefaultMaxSnapshots          = 5
	DefaultMaxWALs               = 5
	DefaultMaxTxnOps             = uint(128)
	DefaultMaxRequestBytes       = 1.5 * 1024 * 1024
	DefaultGRPCKeepAliveMinTime  = 5 * time.Second
	DefaultGRPCKeepAliveInterval = 2 * time.Hour
	DefaultGRPCKeepAliveTimeout  = 20 * time.Second

	DefaultListenPeerURLs   = "http://localhost:2380"
	DefaultListenClientURLs = "http://localhost:2379"

	DefaultLogOutput = "default"

	// DefaultStrictReconfigCheck is the default value for "--strict-reconfig-check" flag.
	// It's enabled by default.
	DefaultStrictReconfigCheck = true
	// DefaultEnableV2 is the default value for "--enable-v2" flag.
	// v2 is enabled by default.
	// TODO: disable v2 when deprecated.
	DefaultEnableV2 = true

	// maxElectionMs specifies the maximum value of election timeout.
	// More details are listed in ../Documentation/tuning.md#time-parameters.
	maxElectionMs = 50000
)

var (
	ErrConflictBootstrapFlags = fmt.Errorf("multiple discovery or bootstrap flags are set. " +
		"Choose one of \"initial-cluster\", \"discovery\" or \"discovery-srv\"")
	ErrUnsetAdvertiseClientURLsFlag = fmt.Errorf("--advertise-client-urls is required when --listen-client-urls is set explicitly")

	DefaultInitialAdvertisePeerURLs = "http://localhost:2380"
	DefaultAdvertiseClientURLs      = "http://localhost:2379"

	defaultHostname   string
	defaultHostStatus error
)

func init() {
	defaultHostname, defaultHostStatus = netutil.GetDefaultHost()
}

// Config holds the arguments for configuring an etcd server.
type Config struct {
	// member

	CorsInfo       *cors.CORSInfo
	LPUrls, LCUrls []url.URL
	Dir            string `json:"data-dir"`
	WalDir         string `json:"wal-dir"`
	MaxSnapFiles   uint   `json:"max-snapshots"`
	MaxWalFiles    uint   `json:"max-wals"`
	Name           string `json:"name"`
	SnapCount      uint64 `json:"snapshot-count"`

	// AutoCompactionMode is either 'periodic' or 'revision'.
	AutoCompactionMode string `json:"auto-compaction-mode"`
	// AutoCompactionRetention is either duration string with time unit
	// (e.g. '5m' for 5-minute), or revision unit (e.g. '5000').
	// If no time unit is provided and compaction mode is 'periodic',
	// the unit defaults to hour. For example, '5' translates into 5-hour.
	AutoCompactionRetention string `json:"auto-compaction-retention"`

	// TickMs is the number of milliseconds between heartbeat ticks.
	// TODO: decouple tickMs and heartbeat tick (current heartbeat tick = 1).
	// make ticks a cluster wide configuration.
	TickMs     uint `json:"heartbeat-interval"`
	ElectionMs uint `json:"election-timeout"`

	// InitialElectionTickAdvance is true, then local member fast-forwards
	// election ticks to speed up "initial" leader election trigger. This
	// benefits the case of larger election ticks. For instance, cross
	// datacenter deployment may require longer election timeout of 10-second.
	// If true, local node does not need wait up to 10-second. Instead,
	// forwards its election ticks to 8-second, and have only 2-second left
	// before leader election.
	//
	// Major assumptions are that:
	//  - cluster has no active leader thus advancing ticks enables faster
	//    leader election, or
	//  - cluster already has an established leader, and rejoining follower
	//    is likely to receive heartbeats from the leader after tick advance
	//    and before election timeout.
	//
	// However, when network from leader to rejoining follower is congested,
	// and the follower does not receive leader heartbeat within left election
	// ticks, disruptive election has to happen thus affecting cluster
	// availabilities.
	//
	// Disabling this would slow down initial bootstrap process for cross
	// datacenter deployments. Make your own tradeoffs by configuring
	// --initial-election-tick-advance at the cost of slow initial bootstrap.
	//
	// If single-node, it advances ticks regardless.
	//
	// See https://github.com/coreos/etcd/issues/9333 for more detail.
	InitialElectionTickAdvance bool `json:"initial-election-tick-advance"`

	QuotaBackendBytes int64 `json:"quota-backend-bytes"`
	MaxTxnOps         uint  `json:"max-txn-ops"`
	MaxRequestBytes   uint  `json:"max-request-bytes"`

	// gRPC server options

	// GRPCKeepAliveMinTime is the minimum interval that a client should
	// wait before pinging server. When client pings "too fast", server
	// sends goaway and closes the connection (errors: too_many_pings,
	// http2.ErrCodeEnhanceYourCalm). When too slow, nothing happens.
	// Server expects client pings only when there is any active streams
	// (PermitWithoutStream is set false).
	GRPCKeepAliveMinTime time.Duration `json:"grpc-keepalive-min-time"`
	// GRPCKeepAliveInterval is the frequency of server-to-client ping
	// to check if a connection is alive. Close a non-responsive connection
	// after an additional duration of Timeout. 0 to disable.
	GRPCKeepAliveInterval time.Duration `json:"grpc-keepalive-interval"`
	// GRPCKeepAliveTimeout is the additional duration of wait
	// before closing a non-responsive connection. 0 to disable.
	GRPCKeepAliveTimeout time.Duration `json:"grpc-keepalive-timeout"`

	// clustering

	APUrls, ACUrls      []url.URL
	ClusterState        string `json:"initial-cluster-state"`
	DNSCluster          string `json:"discovery-srv"`
	Dproxy              string `json:"discovery-proxy"`
	Durl                string `json:"discovery"`
	InitialCluster      string `json:"initial-cluster"`
	InitialClusterToken string `json:"initial-cluster-token"`
	StrictReconfigCheck bool   `json:"strict-reconfig-check"`
	EnableV2            bool   `json:"enable-v2"`

	// security

	ClientTLSInfo transport.TLSInfo
	ClientAutoTLS bool
	PeerTLSInfo   transport.TLSInfo
	PeerAutoTLS   bool

	// CipherSuites is a list of supported TLS cipher suites between
	// client/server and peers. If empty, Go auto-populates the list.
	// Note that cipher suites are prioritized in the given order.
	CipherSuites []string `json:"cipher-suites"`

	// debug

	Debug                 bool   `json:"debug"`
	LogPkgLevels          string `json:"log-package-levels"`
	LogOutput             string `json:"log-output"`
	EnablePprof           bool   `json:"enable-pprof"`
	Metrics               string `json:"metrics"`
	ListenMetricsUrls     []url.URL
	ListenMetricsUrlsJSON string `json:"listen-metrics-urls"`

	// ForceNewCluster starts a new cluster even if previously started; unsafe.
	ForceNewCluster bool `json:"force-new-cluster"`

	// UserHandlers is for registering users handlers and only used for
	// embedding etcd into other applications.
	// The map key is the route path for the handler, and
	// you must ensure it can't be conflicted with etcd's.
	UserHandlers map[string]http.Handler `json:"-"`
	// ServiceRegister is for registering users' gRPC services. A simple usage example:
	//	cfg := embed.NewConfig()
	//	cfg.ServerRegister = func(s *grpc.Server) {
	//		pb.RegisterFooServer(s, &fooServer{})
	//		pb.RegisterBarServer(s, &barServer{})
	//	}
	//	embed.StartEtcd(cfg)
	ServiceRegister func(*grpc.Server) `json:"-"`

	// auth

	AuthToken string `json:"auth-token"`

	// Experimental flags

	ExperimentalInitialCorruptCheck bool          `json:"experimental-initial-corrupt-check"`
	ExperimentalCorruptCheckTime    time.Duration `json:"experimental-corrupt-check-time"`
	ExperimentalEnableV2V3          string        `json:"experimental-enable-v2v3"`
}

// configYAML holds the config suitable for yaml parsing
type configYAML struct {
	Config
	configJSON
}

// configJSON has file options that are translated into Config options
type configJSON struct {
	LPUrlsJSON         string         `json:"listen-peer-urls"`
	LCUrlsJSON         string         `json:"listen-client-urls"`
	CorsJSON           string         `json:"cors"`
	APUrlsJSON         string         `json:"initial-advertise-peer-urls"`
	ACUrlsJSON         string         `json:"advertise-client-urls"`
	ClientSecurityJSON securityConfig `json:"client-transport-security"`
	PeerSecurityJSON   securityConfig `json:"peer-transport-security"`
}

type securityConfig struct {
	CAFile        string `json:"ca-file"`
	CertFile      string `json:"cert-file"`
	KeyFile       string `json:"key-file"`
	CertAuth      bool   `json:"client-cert-auth"`
	TrustedCAFile string `json:"trusted-ca-file"`
	AutoTLS       bool   `json:"auto-tls"`
}

// NewConfig creates a new Config populated with default values.
func NewConfig() *Config {
	lpurl, _ := url.Parse(DefaultListenPeerURLs)
	apurl, _ := url.Parse(DefaultInitialAdvertisePeerURLs)
	lcurl, _ := url.Parse(DefaultListenClientURLs)
	acurl, _ := url.Parse(DefaultAdvertiseClientURLs)
	cfg := &Config{
		CorsInfo:                   &cors.CORSInfo{},
		MaxSnapFiles:               DefaultMaxSnapshots,
		MaxWalFiles:                DefaultMaxWALs,
		Name:                       DefaultName,
		SnapCount:                  etcdserver.DefaultSnapCount,
		MaxTxnOps:                  DefaultMaxTxnOps,
		MaxRequestBytes:            DefaultMaxRequestBytes,
		GRPCKeepAliveMinTime:       DefaultGRPCKeepAliveMinTime,
		GRPCKeepAliveInterval:      DefaultGRPCKeepAliveInterval,
		GRPCKeepAliveTimeout:       DefaultGRPCKeepAliveTimeout,
		TickMs:                     100,
		ElectionMs:                 1000,
		InitialElectionTickAdvance: true,
		LPUrls:                     []url.URL{*lpurl},
		LCUrls:                     []url.URL{*lcurl},
		APUrls:                     []url.URL{*apurl},
		ACUrls:                     []url.URL{*acurl},
		ClusterState:               ClusterStateFlagNew,
		InitialClusterToken:        "etcd-cluster",
		StrictReconfigCheck:        DefaultStrictReconfigCheck,
		LogOutput:                  DefaultLogOutput,
		Metrics:                    "basic",
		EnableV2:                   DefaultEnableV2,
		AuthToken:                  "simple",
	}
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	return cfg
}

func logTLSHandshakeFailure(conn *tls.Conn, err error) {
	state := conn.ConnectionState()
	remoteAddr := conn.RemoteAddr().String()
	serverName := state.ServerName
	if len(state.PeerCertificates) > 0 {
		cert := state.PeerCertificates[0]
		ips, dns := cert.IPAddresses, cert.DNSNames
		plog.Infof("rejected connection from %q (error %q, ServerName %q, IPAddresses %q, DNSNames %q)", remoteAddr, err.Error(), serverName, ips, dns)
	} else {
		plog.Infof("rejected connection from %q (error %q, ServerName %q)", remoteAddr, err.Error(), serverName)
	}
}

// SetupLogging initializes etcd logging.
// Must be called after flag parsing.
func (cfg *Config) SetupLogging() {
	cfg.ClientTLSInfo.HandshakeFailure = logTLSHandshakeFailure
	cfg.PeerTLSInfo.HandshakeFailure = logTLSHandshakeFailure

	capnslog.SetGlobalLogLevel(capnslog.INFO)
	if cfg.Debug {
		capnslog.SetGlobalLogLevel(capnslog.DEBUG)
		grpc.EnableTracing = true
		// enable info, warning, error
		grpclog.SetLoggerV2(grpclog.NewLoggerV2(os.Stderr, os.Stderr, os.Stderr))
	} else {
		// only discard info
		grpclog.SetLoggerV2(grpclog.NewLoggerV2(ioutil.Discard, os.Stderr, os.Stderr))
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
	switch cfg.LogOutput {
	case "stdout":
		capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stdout, cfg.Debug))
	case "stderr":
		capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stderr, cfg.Debug))
	case DefaultLogOutput:
	default:
		plog.Panicf(`unknown log-output %q (only supports %q, "stdout", "stderr")`, cfg.LogOutput, DefaultLogOutput)
	}
}

func ConfigFromFile(path string) (*Config, error) {
	cfg := &configYAML{Config: *NewConfig()}
	if err := cfg.configFromFile(path); err != nil {
		return nil, err
	}
	return &cfg.Config, nil
}

func (cfg *configYAML) configFromFile(path string) error {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	defaultInitialCluster := cfg.InitialCluster

	err = yaml.Unmarshal(b, cfg)
	if err != nil {
		return err
	}

	if cfg.LPUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.LPUrlsJSON, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up listen-peer-urls: %v", err)
		}
		cfg.LPUrls = []url.URL(u)
	}

	if cfg.LCUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.LCUrlsJSON, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up listen-client-urls: %v", err)
		}
		cfg.LCUrls = []url.URL(u)
	}

	if cfg.CorsJSON != "" {
		if err := cfg.CorsInfo.Set(cfg.CorsJSON); err != nil {
			plog.Panicf("unexpected error setting up cors: %v", err)
		}
	}

	if cfg.APUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.APUrlsJSON, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up initial-advertise-peer-urls: %v", err)
		}
		cfg.APUrls = []url.URL(u)
	}

	if cfg.ACUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.ACUrlsJSON, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up advertise-peer-urls: %v", err)
		}
		cfg.ACUrls = []url.URL(u)
	}

	if cfg.ListenMetricsUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.ListenMetricsUrlsJSON, ","))
		if err != nil {
			plog.Fatalf("unexpected error setting up listen-metrics-urls: %v", err)
		}
		cfg.ListenMetricsUrls = []url.URL(u)
	}

	// If a discovery flag is set, clear default initial cluster set by InitialClusterFromName
	if (cfg.Durl != "" || cfg.DNSCluster != "") && cfg.InitialCluster == defaultInitialCluster {
		cfg.InitialCluster = ""
	}
	if cfg.ClusterState == "" {
		cfg.ClusterState = ClusterStateFlagNew
	}

	copySecurityDetails := func(tls *transport.TLSInfo, ysc *securityConfig) {
		tls.CAFile = ysc.CAFile
		tls.CertFile = ysc.CertFile
		tls.KeyFile = ysc.KeyFile
		tls.ClientCertAuth = ysc.CertAuth
		tls.TrustedCAFile = ysc.TrustedCAFile
	}
	copySecurityDetails(&cfg.ClientTLSInfo, &cfg.ClientSecurityJSON)
	copySecurityDetails(&cfg.PeerTLSInfo, &cfg.PeerSecurityJSON)
	cfg.ClientAutoTLS = cfg.ClientSecurityJSON.AutoTLS
	cfg.PeerAutoTLS = cfg.PeerSecurityJSON.AutoTLS

	return cfg.Validate()
}

func updateCipherSuites(tls *transport.TLSInfo, ss []string) error {
	if len(tls.CipherSuites) > 0 && len(ss) > 0 {
		return fmt.Errorf("TLSInfo.CipherSuites is already specified (given %v)", ss)
	}
	if len(ss) > 0 {
		cs := make([]uint16, len(ss))
		for i, s := range ss {
			var ok bool
			cs[i], ok = tlsutil.GetCipherSuite(s)
			if !ok {
				return fmt.Errorf("unexpected TLS cipher suite %q", s)
			}
		}
		tls.CipherSuites = cs
	}
	return nil
}

// Validate ensures that '*embed.Config' fields are properly configured.
func (cfg *Config) Validate() error {
	if err := checkBindURLs(cfg.LPUrls); err != nil {
		return err
	}
	if err := checkBindURLs(cfg.LCUrls); err != nil {
		return err
	}
	if err := checkBindURLs(cfg.ListenMetricsUrls); err != nil {
		return err
	}
	if err := checkHostURLs(cfg.APUrls); err != nil {
		// TODO: return err in v3.4
		addrs := make([]string, len(cfg.APUrls))
		for i := range cfg.APUrls {
			addrs[i] = cfg.APUrls[i].String()
		}
		plog.Warningf("advertise-peer-urls %q is deprecated (%v)", strings.Join(addrs, ","), err)
	}
	if err := checkHostURLs(cfg.ACUrls); err != nil {
		// TODO: return err in v3.4
		addrs := make([]string, len(cfg.ACUrls))
		for i := range cfg.ACUrls {
			addrs[i] = cfg.ACUrls[i].String()
		}
		plog.Warningf("advertise-client-urls %q is deprecated (%v)", strings.Join(addrs, ","), err)
	}

	// Check if conflicting flags are passed.
	nSet := 0
	for _, v := range []bool{cfg.Durl != "", cfg.InitialCluster != "", cfg.DNSCluster != ""} {
		if v {
			nSet++
		}
	}

	if cfg.ClusterState != ClusterStateFlagNew && cfg.ClusterState != ClusterStateFlagExisting {
		return fmt.Errorf("unexpected clusterState %q", cfg.ClusterState)
	}

	if nSet > 1 {
		return ErrConflictBootstrapFlags
	}

	if cfg.TickMs <= 0 {
		return fmt.Errorf("--heartbeat-interval must be >0 (set to %dms)", cfg.TickMs)
	}
	if cfg.ElectionMs <= 0 {
		return fmt.Errorf("--election-timeout must be >0 (set to %dms)", cfg.ElectionMs)
	}
	if 5*cfg.TickMs > cfg.ElectionMs {
		return fmt.Errorf("--election-timeout[%vms] should be at least as 5 times as --heartbeat-interval[%vms]", cfg.ElectionMs, cfg.TickMs)
	}
	if cfg.ElectionMs > maxElectionMs {
		return fmt.Errorf("--election-timeout[%vms] is too long, and should be set less than %vms", cfg.ElectionMs, maxElectionMs)
	}

	// check this last since proxying in etcdmain may make this OK
	if cfg.LCUrls != nil && cfg.ACUrls == nil {
		return ErrUnsetAdvertiseClientURLsFlag
	}

	switch cfg.AutoCompactionMode {
	case "":
	case compactor.ModeRevision, compactor.ModePeriodic:
	default:
		return fmt.Errorf("unknown auto-compaction-mode %q", cfg.AutoCompactionMode)
	}

	return nil
}

// PeerURLsMapAndToken sets up an initial peer URLsMap and cluster token for bootstrap or discovery.
func (cfg *Config) PeerURLsMapAndToken(which string) (urlsmap types.URLsMap, token string, err error) {
	token = cfg.InitialClusterToken
	switch {
	case cfg.Durl != "":
		urlsmap = types.URLsMap{}
		// If using discovery, generate a temporary cluster based on
		// self's advertised peer URLs
		urlsmap[cfg.Name] = cfg.APUrls
		token = cfg.Durl
	case cfg.DNSCluster != "":
		clusterStrs, cerr := srv.GetCluster("etcd-server", cfg.Name, cfg.DNSCluster, cfg.APUrls)
		if cerr != nil {
			plog.Errorf("couldn't resolve during SRV discovery (%v)", cerr)
			return nil, "", cerr
		}
		for _, s := range clusterStrs {
			plog.Noticef("got bootstrap from DNS for etcd-server at %s", s)
		}
		clusterStr := strings.Join(clusterStrs, ",")
		if strings.Contains(clusterStr, "https://") && cfg.PeerTLSInfo.CAFile == "" {
			cfg.PeerTLSInfo.ServerName = cfg.DNSCluster
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
	}
	return urlsmap, token, err
}

func (cfg Config) InitialClusterFromName(name string) (ret string) {
	if len(cfg.APUrls) == 0 {
		return ""
	}
	n := name
	if name == "" {
		n = DefaultName
	}
	for i := range cfg.APUrls {
		ret = ret + "," + n + "=" + cfg.APUrls[i].String()
	}
	return ret[1:]
}

func (cfg Config) IsNewCluster() bool { return cfg.ClusterState == ClusterStateFlagNew }
func (cfg Config) ElectionTicks() int { return int(cfg.ElectionMs / cfg.TickMs) }

func (cfg Config) defaultPeerHost() bool {
	return len(cfg.APUrls) == 1 && cfg.APUrls[0].String() == DefaultInitialAdvertisePeerURLs
}

func (cfg Config) defaultClientHost() bool {
	return len(cfg.ACUrls) == 1 && cfg.ACUrls[0].String() == DefaultAdvertiseClientURLs
}

func (cfg *Config) ClientSelfCert() (err error) {
	if !cfg.ClientAutoTLS {
		return nil
	}
	if !cfg.ClientTLSInfo.Empty() {
		plog.Warningf("ignoring client auto TLS since certs given")
		return nil
	}
	chosts := make([]string, len(cfg.LCUrls))
	for i, u := range cfg.LCUrls {
		chosts[i] = u.Host
	}
	cfg.ClientTLSInfo, err = transport.SelfCert(filepath.Join(cfg.Dir, "fixtures", "client"), chosts)
	if err != nil {
		return err
	}
	return updateCipherSuites(&cfg.ClientTLSInfo, cfg.CipherSuites)
}

func (cfg *Config) PeerSelfCert() (err error) {
	if !cfg.PeerAutoTLS {
		return nil
	}
	if !cfg.PeerTLSInfo.Empty() {
		plog.Warningf("ignoring peer auto TLS since certs given")
		return nil
	}
	phosts := make([]string, len(cfg.LPUrls))
	for i, u := range cfg.LPUrls {
		phosts[i] = u.Host
	}
	cfg.PeerTLSInfo, err = transport.SelfCert(filepath.Join(cfg.Dir, "fixtures", "peer"), phosts)
	if err != nil {
		return err
	}
	return updateCipherSuites(&cfg.PeerTLSInfo, cfg.CipherSuites)
}

// UpdateDefaultClusterFromName updates cluster advertise URLs with, if available, default host,
// if advertise URLs are default values(localhost:2379,2380) AND if listen URL is 0.0.0.0.
// e.g. advertise peer URL localhost:2380 or listen peer URL 0.0.0.0:2380
// then the advertise peer host would be updated with machine's default host,
// while keeping the listen URL's port.
// User can work around this by explicitly setting URL with 127.0.0.1.
// It returns the default hostname, if used, and the error, if any, from getting the machine's default host.
// TODO: check whether fields are set instead of whether fields have default value
func (cfg *Config) UpdateDefaultClusterFromName(defaultInitialCluster string) (string, error) {
	if defaultHostname == "" || defaultHostStatus != nil {
		// update 'initial-cluster' when only the name is specified (e.g. 'etcd --name=abc')
		if cfg.Name != DefaultName && cfg.InitialCluster == defaultInitialCluster {
			cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
		}
		return "", defaultHostStatus
	}

	used := false
	pip, pport := cfg.LPUrls[0].Hostname(), cfg.LPUrls[0].Port()
	if cfg.defaultPeerHost() && pip == "0.0.0.0" {
		cfg.APUrls[0] = url.URL{Scheme: cfg.APUrls[0].Scheme, Host: fmt.Sprintf("%s:%s", defaultHostname, pport)}
		used = true
	}
	// update 'initial-cluster' when only the name is specified (e.g. 'etcd --name=abc')
	if cfg.Name != DefaultName && cfg.InitialCluster == defaultInitialCluster {
		cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	}

	cip, cport := cfg.LCUrls[0].Hostname(), cfg.LCUrls[0].Port()
	if cfg.defaultClientHost() && cip == "0.0.0.0" {
		cfg.ACUrls[0] = url.URL{Scheme: cfg.ACUrls[0].Scheme, Host: fmt.Sprintf("%s:%s", defaultHostname, cport)}
		used = true
	}
	dhost := defaultHostname
	if !used {
		dhost = ""
	}
	return dhost, defaultHostStatus
}

// checkBindURLs returns an error if any URL uses a domain name.
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
			return fmt.Errorf("expected IP in URL for binding (%s)", url.String())
		}
	}
	return nil
}

func checkHostURLs(urls []url.URL) error {
	for _, url := range urls {
		host, _, err := net.SplitHostPort(url.Host)
		if err != nil {
			return err
		}
		if host == "" {
			return fmt.Errorf("unexpected empty host (%s)", url.String())
		}
	}
	return nil
}
