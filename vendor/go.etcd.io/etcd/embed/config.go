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
	"sync"
	"time"

	"go.etcd.io/etcd/etcdserver"
	"go.etcd.io/etcd/etcdserver/api/v3compactor"
	"go.etcd.io/etcd/pkg/flags"
	"go.etcd.io/etcd/pkg/logutil"
	"go.etcd.io/etcd/pkg/netutil"
	"go.etcd.io/etcd/pkg/srv"
	"go.etcd.io/etcd/pkg/tlsutil"
	"go.etcd.io/etcd/pkg/transport"
	"go.etcd.io/etcd/pkg/types"

	bolt "go.etcd.io/bbolt"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc"
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
	JournalLogOutput = "systemd/journal"
	StdErrLogOutput  = "stderr"
	StdOutLogOutput  = "stdout"

	// DefaultStrictReconfigCheck is the default value for "--strict-reconfig-check" flag.
	// It's enabled by default.
	DefaultStrictReconfigCheck = true
	// DefaultEnableV2 is the default value for "--enable-v2" flag.
	// v2 API is disabled by default.
	DefaultEnableV2 = false

	// maxElectionMs specifies the maximum value of election timeout.
	// More details are listed in ../Documentation/tuning.md#time-parameters.
	maxElectionMs = 50000
	// backend freelist map type
	freelistMapType = "map"
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

var (
	// CompactorModePeriodic is periodic compaction mode
	// for "Config.AutoCompactionMode" field.
	// If "AutoCompactionMode" is CompactorModePeriodic and
	// "AutoCompactionRetention" is "1h", it automatically compacts
	// compacts storage every hour.
	CompactorModePeriodic = v3compactor.ModePeriodic

	// CompactorModeRevision is revision-based compaction mode
	// for "Config.AutoCompactionMode" field.
	// If "AutoCompactionMode" is CompactorModeRevision and
	// "AutoCompactionRetention" is "1000", it compacts log on
	// revision 5000 when the current revision is 6000.
	// This runs every 5-minute if enough of logs have proceeded.
	CompactorModeRevision = v3compactor.ModeRevision
)

func init() {
	defaultHostname, defaultHostStatus = netutil.GetDefaultHost()
}

// Config holds the arguments for configuring an etcd server.
type Config struct {
	Name   string `json:"name"`
	Dir    string `json:"data-dir"`
	WalDir string `json:"wal-dir"`

	SnapshotCount uint64 `json:"snapshot-count"`

	// SnapshotCatchUpEntries is the number of entries for a slow follower
	// to catch-up after compacting the raft storage entries.
	// We expect the follower has a millisecond level latency with the leader.
	// The max throughput is around 10K. Keep a 5K entries is enough for helping
	// follower to catch up.
	// WARNING: only change this for tests.
	// Always use "DefaultSnapshotCatchUpEntries"
	SnapshotCatchUpEntries uint64

	MaxSnapFiles uint `json:"max-snapshots"`
	MaxWalFiles  uint `json:"max-wals"`

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
	// See https://github.com/etcd-io/etcd/issues/9333 for more detail.
	InitialElectionTickAdvance bool `json:"initial-election-tick-advance"`

	// BackendBatchInterval is the maximum time before commit the backend transaction.
	BackendBatchInterval time.Duration `json:"backend-batch-interval"`
	// BackendBatchLimit is the maximum operations before commit the backend transaction.
	BackendBatchLimit int   `json:"backend-batch-limit"`
	QuotaBackendBytes int64 `json:"quota-backend-bytes"`
	MaxTxnOps         uint  `json:"max-txn-ops"`
	MaxRequestBytes   uint  `json:"max-request-bytes"`

	LPUrls, LCUrls []url.URL
	APUrls, ACUrls []url.URL
	ClientTLSInfo  transport.TLSInfo
	ClientAutoTLS  bool
	PeerTLSInfo    transport.TLSInfo
	PeerAutoTLS    bool

	// CipherSuites is a list of supported TLS cipher suites between
	// client/server and peers. If empty, Go auto-populates the list.
	// Note that cipher suites are prioritized in the given order.
	CipherSuites []string `json:"cipher-suites"`

	ClusterState          string `json:"initial-cluster-state"`
	DNSCluster            string `json:"discovery-srv"`
	DNSClusterServiceName string `json:"discovery-srv-name"`
	Dproxy                string `json:"discovery-proxy"`
	Durl                  string `json:"discovery"`
	InitialCluster        string `json:"initial-cluster"`
	InitialClusterToken   string `json:"initial-cluster-token"`
	StrictReconfigCheck   bool   `json:"strict-reconfig-check"`
	EnableV2              bool   `json:"enable-v2"`

	// AutoCompactionMode is either 'periodic' or 'revision'.
	AutoCompactionMode string `json:"auto-compaction-mode"`
	// AutoCompactionRetention is either duration string with time unit
	// (e.g. '5m' for 5-minute), or revision unit (e.g. '5000').
	// If no time unit is provided and compaction mode is 'periodic',
	// the unit defaults to hour. For example, '5' translates into 5-hour.
	AutoCompactionRetention string `json:"auto-compaction-retention"`

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

	// PreVote is true to enable Raft Pre-Vote.
	// If enabled, Raft runs an additional election phase
	// to check whether it would get enough votes to win
	// an election, thus minimizing disruptions.
	// TODO: enable by default in 3.5.
	PreVote bool `json:"pre-vote"`

	CORS map[string]struct{}

	// HostWhitelist lists acceptable hostnames from HTTP client requests.
	// Client origin policy protects against "DNS Rebinding" attacks
	// to insecure etcd servers. That is, any website can simply create
	// an authorized DNS name, and direct DNS to "localhost" (or any
	// other address). Then, all HTTP endpoints of etcd server listening
	// on "localhost" becomes accessible, thus vulnerable to DNS rebinding
	// attacks. See "CVE-2018-5702" for more detail.
	//
	// 1. If client connection is secure via HTTPS, allow any hostnames.
	// 2. If client connection is not secure and "HostWhitelist" is not empty,
	//    only allow HTTP requests whose Host field is listed in whitelist.
	//
	// Note that the client origin policy is enforced whether authentication
	// is enabled or not, for tighter controls.
	//
	// By default, "HostWhitelist" is "*", which allows any hostnames.
	// Note that when specifying hostnames, loopback addresses are not added
	// automatically. To allow loopback interfaces, leave it empty or set it "*",
	// or add them to whitelist manually (e.g. "localhost", "127.0.0.1", etc.).
	//
	// CVE-2018-5702 reference:
	// - https://bugs.chromium.org/p/project-zero/issues/detail?id=1447#c2
	// - https://github.com/transmission/transmission/pull/468
	// - https://github.com/etcd-io/etcd/issues/9353
	HostWhitelist map[string]struct{}

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

	AuthToken  string `json:"auth-token"`
	BcryptCost uint   `json:"bcrypt-cost"`

	ExperimentalInitialCorruptCheck bool          `json:"experimental-initial-corrupt-check"`
	ExperimentalCorruptCheckTime    time.Duration `json:"experimental-corrupt-check-time"`
	ExperimentalEnableV2V3          string        `json:"experimental-enable-v2v3"`
	// ExperimentalBackendFreelistType specifies the type of freelist that boltdb backend uses (array and map are supported types).
	ExperimentalBackendFreelistType string `json:"experimental-backend-bbolt-freelist-type"`
	// ExperimentalEnableLeaseCheckpoint enables primary lessor to persist lease remainingTTL to prevent indefinite auto-renewal of long lived leases.
	ExperimentalEnableLeaseCheckpoint bool `json:"experimental-enable-lease-checkpoint"`
	ExperimentalCompactionBatchLimit  int  `json:"experimental-compaction-batch-limit"`

	// ForceNewCluster starts a new cluster even if previously started; unsafe.
	ForceNewCluster bool `json:"force-new-cluster"`

	EnablePprof           bool   `json:"enable-pprof"`
	Metrics               string `json:"metrics"`
	ListenMetricsUrls     []url.URL
	ListenMetricsUrlsJSON string `json:"listen-metrics-urls"`

	// Logger is logger options: "zap", "capnslog".
	// WARN: "capnslog" is being deprecated in v3.5.
	Logger string `json:"logger"`
	// LogLevel configures log level. Only supports debug, info, warn, error, panic, or fatal. Default 'info'.
	LogLevel string `json:"log-level"`
	// LogOutputs is either:
	//  - "default" as os.Stderr,
	//  - "stderr" as os.Stderr,
	//  - "stdout" as os.Stdout,
	//  - file path to append server logs to.
	// It can be multiple when "Logger" is zap.
	LogOutputs []string `json:"log-outputs"`

	// ZapLoggerBuilder is used to build the zap logger.
	ZapLoggerBuilder func(*Config) error

	// logger logs server-side operations. The default is nil,
	// and "setupLogging" must be called before starting server.
	// Do not set logger directly.
	loggerMu *sync.RWMutex
	logger   *zap.Logger

	// loggerConfig is server logger configuration for Raft logger.
	// Must be either: "loggerConfig != nil" or "loggerCore != nil && loggerWriteSyncer != nil".
	loggerConfig *zap.Config
	// loggerCore is "zapcore.Core" for raft logger.
	// Must be either: "loggerConfig != nil" or "loggerCore != nil && loggerWriteSyncer != nil".
	loggerCore        zapcore.Core
	loggerWriteSyncer zapcore.WriteSyncer

	// EnableGRPCGateway is false to disable grpc gateway.
	EnableGRPCGateway bool `json:"enable-grpc-gateway"`

	// TO BE DEPRECATED

	// DeprecatedLogOutput is to be deprecated in v3.5.
	// Just here for safe migration in v3.4.
	DeprecatedLogOutput []string `json:"log-output"`
	// Debug is true, to enable debug level logging.
	// WARNING: to be deprecated in 3.5. Use "--log-level=debug" instead.
	Debug bool `json:"debug"`
	// LogPkgLevels is being deprecated in v3.5.
	// Only valid if "logger" option is "capnslog".
	// WARN: DO NOT USE THIS!
	LogPkgLevels string `json:"log-package-levels"`
}

// configYAML holds the config suitable for yaml parsing
type configYAML struct {
	Config
	configJSON
}

// configJSON has file options that are translated into Config options
type configJSON struct {
	LPUrlsJSON string `json:"listen-peer-urls"`
	LCUrlsJSON string `json:"listen-client-urls"`
	APUrlsJSON string `json:"initial-advertise-peer-urls"`
	ACUrlsJSON string `json:"advertise-client-urls"`

	CORSJSON          string `json:"cors"`
	HostWhitelistJSON string `json:"host-whitelist"`

	ClientSecurityJSON securityConfig `json:"client-transport-security"`
	PeerSecurityJSON   securityConfig `json:"peer-transport-security"`
}

type securityConfig struct {
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
		MaxSnapFiles: DefaultMaxSnapshots,
		MaxWalFiles:  DefaultMaxWALs,

		Name: DefaultName,

		SnapshotCount:          etcdserver.DefaultSnapshotCount,
		SnapshotCatchUpEntries: etcdserver.DefaultSnapshotCatchUpEntries,

		MaxTxnOps:       DefaultMaxTxnOps,
		MaxRequestBytes: DefaultMaxRequestBytes,

		GRPCKeepAliveMinTime:  DefaultGRPCKeepAliveMinTime,
		GRPCKeepAliveInterval: DefaultGRPCKeepAliveInterval,
		GRPCKeepAliveTimeout:  DefaultGRPCKeepAliveTimeout,

		TickMs:                     100,
		ElectionMs:                 1000,
		InitialElectionTickAdvance: true,

		LPUrls: []url.URL{*lpurl},
		LCUrls: []url.URL{*lcurl},
		APUrls: []url.URL{*apurl},
		ACUrls: []url.URL{*acurl},

		ClusterState:        ClusterStateFlagNew,
		InitialClusterToken: "etcd-cluster",

		StrictReconfigCheck: DefaultStrictReconfigCheck,
		Metrics:             "basic",
		EnableV2:            DefaultEnableV2,

		CORS:          map[string]struct{}{"*": {}},
		HostWhitelist: map[string]struct{}{"*": {}},

		AuthToken:  "simple",
		BcryptCost: uint(bcrypt.DefaultCost),

		PreVote: false, // TODO: enable by default in v3.5

		loggerMu:            new(sync.RWMutex),
		logger:              nil,
		Logger:              "capnslog",
		DeprecatedLogOutput: []string{DefaultLogOutput},
		LogOutputs:          []string{DefaultLogOutput},
		Debug:               false,
		LogLevel:            logutil.DefaultLogLevel,
		LogPkgLevels:        "",
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
			fmt.Fprintf(os.Stderr, "unexpected error setting up listen-peer-urls: %v\n", err)
			os.Exit(1)
		}
		cfg.LPUrls = []url.URL(u)
	}

	if cfg.LCUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.LCUrlsJSON, ","))
		if err != nil {
			fmt.Fprintf(os.Stderr, "unexpected error setting up listen-client-urls: %v\n", err)
			os.Exit(1)
		}
		cfg.LCUrls = []url.URL(u)
	}

	if cfg.APUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.APUrlsJSON, ","))
		if err != nil {
			fmt.Fprintf(os.Stderr, "unexpected error setting up initial-advertise-peer-urls: %v\n", err)
			os.Exit(1)
		}
		cfg.APUrls = []url.URL(u)
	}

	if cfg.ACUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.ACUrlsJSON, ","))
		if err != nil {
			fmt.Fprintf(os.Stderr, "unexpected error setting up advertise-peer-urls: %v\n", err)
			os.Exit(1)
		}
		cfg.ACUrls = []url.URL(u)
	}

	if cfg.ListenMetricsUrlsJSON != "" {
		u, err := types.NewURLs(strings.Split(cfg.ListenMetricsUrlsJSON, ","))
		if err != nil {
			fmt.Fprintf(os.Stderr, "unexpected error setting up listen-metrics-urls: %v\n", err)
			os.Exit(1)
		}
		cfg.ListenMetricsUrls = []url.URL(u)
	}

	if cfg.CORSJSON != "" {
		uv := flags.NewUniqueURLsWithExceptions(cfg.CORSJSON, "*")
		cfg.CORS = uv.Values
	}

	if cfg.HostWhitelistJSON != "" {
		uv := flags.NewUniqueStringsValue(cfg.HostWhitelistJSON)
		cfg.HostWhitelist = uv.Values
	}

	// If a discovery flag is set, clear default initial cluster set by InitialClusterFromName
	if (cfg.Durl != "" || cfg.DNSCluster != "") && cfg.InitialCluster == defaultInitialCluster {
		cfg.InitialCluster = ""
	}
	if cfg.ClusterState == "" {
		cfg.ClusterState = ClusterStateFlagNew
	}

	copySecurityDetails := func(tls *transport.TLSInfo, ysc *securityConfig) {
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
	if err := cfg.setupLogging(); err != nil {
		return err
	}
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
		addrs := cfg.getAPURLs()
		return fmt.Errorf(`--initial-advertise-peer-urls %q must be "host:port" (%v)`, strings.Join(addrs, ","), err)
	}
	if err := checkHostURLs(cfg.ACUrls); err != nil {
		addrs := cfg.getACURLs()
		return fmt.Errorf(`--advertise-client-urls %q must be "host:port" (%v)`, strings.Join(addrs, ","), err)
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
	case CompactorModeRevision, CompactorModePeriodic:
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
		clusterStrs, cerr := cfg.GetDNSClusterNames()
		lg := cfg.logger
		if cerr != nil {
			if lg != nil {
				lg.Warn("failed to resolve during SRV discovery", zap.Error(cerr))
			} else {
				plog.Errorf("couldn't resolve during SRV discovery (%v)", cerr)
			}
			return nil, "", cerr
		}
		for _, s := range clusterStrs {
			if lg != nil {
				lg.Info("got bootstrap from DNS for etcd-server", zap.String("node", s))
			} else {
				plog.Noticef("got bootstrap from DNS for etcd-server at %s", s)
			}
		}
		clusterStr := strings.Join(clusterStrs, ",")
		if strings.Contains(clusterStr, "https://") && cfg.PeerTLSInfo.TrustedCAFile == "" {
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

// GetDNSClusterNames uses DNS SRV records to get a list of initial nodes for cluster bootstrapping.
func (cfg *Config) GetDNSClusterNames() ([]string, error) {
	var (
		clusterStrs       []string
		cerr              error
		serviceNameSuffix string
	)
	if cfg.DNSClusterServiceName != "" {
		serviceNameSuffix = "-" + cfg.DNSClusterServiceName
	}

	lg := cfg.GetLogger()

	// Use both etcd-server-ssl and etcd-server for discovery.
	// Combine the results if both are available.
	clusterStrs, cerr = srv.GetCluster("https", "etcd-server-ssl"+serviceNameSuffix, cfg.Name, cfg.DNSCluster, cfg.APUrls)
	if cerr != nil {
		clusterStrs = make([]string, 0)
	}
	if lg != nil {
		lg.Info(
			"get cluster for etcd-server-ssl SRV",
			zap.String("service-scheme", "https"),
			zap.String("service-name", "etcd-server-ssl"+serviceNameSuffix),
			zap.String("server-name", cfg.Name),
			zap.String("discovery-srv", cfg.DNSCluster),
			zap.Strings("advertise-peer-urls", cfg.getAPURLs()),
			zap.Strings("found-cluster", clusterStrs),
			zap.Error(cerr),
		)
	}

	defaultHTTPClusterStrs, httpCerr := srv.GetCluster("http", "etcd-server"+serviceNameSuffix, cfg.Name, cfg.DNSCluster, cfg.APUrls)
	if httpCerr != nil {
		clusterStrs = append(clusterStrs, defaultHTTPClusterStrs...)
	}
	if lg != nil {
		lg.Info(
			"get cluster for etcd-server SRV",
			zap.String("service-scheme", "http"),
			zap.String("service-name", "etcd-server"+serviceNameSuffix),
			zap.String("server-name", cfg.Name),
			zap.String("discovery-srv", cfg.DNSCluster),
			zap.Strings("advertise-peer-urls", cfg.getAPURLs()),
			zap.Strings("found-cluster", clusterStrs),
			zap.Error(httpCerr),
		)
	}

	return clusterStrs, cerr
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
		if cfg.logger != nil {
			cfg.logger.Warn("ignoring client auto TLS since certs given")
		} else {
			plog.Warningf("ignoring client auto TLS since certs given")
		}
		return nil
	}
	chosts := make([]string, len(cfg.LCUrls))
	for i, u := range cfg.LCUrls {
		chosts[i] = u.Host
	}
	cfg.ClientTLSInfo, err = transport.SelfCert(cfg.logger, filepath.Join(cfg.Dir, "fixtures", "client"), chosts)
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
		if cfg.logger != nil {
			cfg.logger.Warn("ignoring peer auto TLS since certs given")
		} else {
			plog.Warningf("ignoring peer auto TLS since certs given")
		}
		return nil
	}
	phosts := make([]string, len(cfg.LPUrls))
	for i, u := range cfg.LPUrls {
		phosts[i] = u.Host
	}
	cfg.PeerTLSInfo, err = transport.SelfCert(cfg.logger, filepath.Join(cfg.Dir, "fixtures", "peer"), phosts)
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

func (cfg *Config) getAPURLs() (ss []string) {
	ss = make([]string, len(cfg.APUrls))
	for i := range cfg.APUrls {
		ss[i] = cfg.APUrls[i].String()
	}
	return ss
}

func (cfg *Config) getLPURLs() (ss []string) {
	ss = make([]string, len(cfg.LPUrls))
	for i := range cfg.LPUrls {
		ss[i] = cfg.LPUrls[i].String()
	}
	return ss
}

func (cfg *Config) getACURLs() (ss []string) {
	ss = make([]string, len(cfg.ACUrls))
	for i := range cfg.ACUrls {
		ss[i] = cfg.ACUrls[i].String()
	}
	return ss
}

func (cfg *Config) getLCURLs() (ss []string) {
	ss = make([]string, len(cfg.LCUrls))
	for i := range cfg.LCUrls {
		ss[i] = cfg.LCUrls[i].String()
	}
	return ss
}

func (cfg *Config) getMetricsURLs() (ss []string) {
	ss = make([]string, len(cfg.ListenMetricsUrls))
	for i := range cfg.ListenMetricsUrls {
		ss[i] = cfg.ListenMetricsUrls[i].String()
	}
	return ss
}

func parseBackendFreelistType(freelistType string) bolt.FreelistType {
	if freelistType == freelistMapType {
		return bolt.FreelistMapType
	}

	return bolt.FreelistArrayType
}
