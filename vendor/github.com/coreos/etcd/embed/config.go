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
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/coreos/etcd/discovery"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/cors"
	"github.com/coreos/etcd/pkg/netutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/ghodss/yaml"
)

const (
	ClusterStateFlagNew      = "new"
	ClusterStateFlagExisting = "existing"

	DefaultName         = "default"
	DefaultMaxSnapshots = 5
	DefaultMaxWALs      = 5

	DefaultListenPeerURLs   = "http://localhost:2380"
	DefaultListenClientURLs = "http://localhost:2379"

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

	CorsInfo                *cors.CORSInfo
	LPUrls, LCUrls          []url.URL
	Dir                     string `json:"data-dir"`
	WalDir                  string `json:"wal-dir"`
	MaxSnapFiles            uint   `json:"max-snapshots"`
	MaxWalFiles             uint   `json:"max-wals"`
	Name                    string `json:"name"`
	SnapCount               uint64 `json:"snapshot-count"`
	AutoCompactionRetention int    `json:"auto-compaction-retention"`

	// TickMs is the number of milliseconds between heartbeat ticks.
	// TODO: decouple tickMs and heartbeat tick (current heartbeat tick = 1).
	// make ticks a cluster wide configuration.
	TickMs            uint  `json:"heartbeat-interval"`
	ElectionMs        uint  `json:"election-timeout"`
	QuotaBackendBytes int64 `json:"quota-backend-bytes"`

	// clustering

	APUrls, ACUrls      []url.URL
	ClusterState        string `json:"initial-cluster-state"`
	DNSCluster          string `json:"discovery-srv"`
	Dproxy              string `json:"discovery-proxy"`
	Durl                string `json:"discovery"`
	InitialCluster      string `json:"initial-cluster"`
	InitialClusterToken string `json:"initial-cluster-token"`
	StrictReconfigCheck bool   `json:"strict-reconfig-check"`

	// security

	ClientTLSInfo transport.TLSInfo
	ClientAutoTLS bool
	PeerTLSInfo   transport.TLSInfo
	PeerAutoTLS   bool

	// debug

	Debug        bool   `json:"debug"`
	LogPkgLevels string `json:"log-package-levels"`
	EnablePprof  bool
	Metrics      string `json:"metrics"`

	// ForceNewCluster starts a new cluster even if previously started; unsafe.
	ForceNewCluster bool `json:"force-new-cluster"`

	// UserHandlers is for registering users handlers and only used for
	// embedding etcd into other applications.
	// The map key is the route path for the handler, and
	// you must ensure it can't be conflicted with etcd's.
	UserHandlers map[string]http.Handler `json:"-"`
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
		CorsInfo:            &cors.CORSInfo{},
		MaxSnapFiles:        DefaultMaxSnapshots,
		MaxWalFiles:         DefaultMaxWALs,
		Name:                DefaultName,
		SnapCount:           etcdserver.DefaultSnapCount,
		TickMs:              100,
		ElectionMs:          1000,
		LPUrls:              []url.URL{*lpurl},
		LCUrls:              []url.URL{*lcurl},
		APUrls:              []url.URL{*apurl},
		ACUrls:              []url.URL{*acurl},
		ClusterState:        ClusterStateFlagNew,
		InitialClusterToken: "etcd-cluster",
		StrictReconfigCheck: true,
		Metrics:             "basic",
	}
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	return cfg
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

	if (cfg.Durl != "" || cfg.DNSCluster != "") && cfg.InitialCluster == cfg.InitialClusterFromName(cfg.Name) {
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

func (cfg *Config) Validate() error {
	if err := checkBindURLs(cfg.LPUrls); err != nil {
		return err
	}
	if err := checkBindURLs(cfg.LCUrls); err != nil {
		return err
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

	return nil
}

// PeerURLsMapAndToken sets up an initial peer URLsMap and cluster token for bootstrap or discovery.
func (cfg *Config) PeerURLsMapAndToken(which string) (urlsmap types.URLsMap, token string, err error) {
	switch {
	case cfg.Durl != "":
		urlsmap = types.URLsMap{}
		// If using discovery, generate a temporary cluster based on
		// self's advertised peer URLs
		urlsmap[cfg.Name] = cfg.APUrls
		token = cfg.Durl
	case cfg.DNSCluster != "":
		var clusterStr string
		clusterStr, token, err = discovery.SRVGetCluster(cfg.Name, cfg.DNSCluster, cfg.InitialClusterToken, cfg.APUrls)
		if err != nil {
			return nil, "", err
		}
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
		token = cfg.InitialClusterToken
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
	pip, pport, _ := net.SplitHostPort(cfg.LPUrls[0].Host)
	if cfg.defaultPeerHost() && pip == "0.0.0.0" {
		cfg.APUrls[0] = url.URL{Scheme: cfg.APUrls[0].Scheme, Host: fmt.Sprintf("%s:%s", defaultHostname, pport)}
		used = true
	}
	// update 'initial-cluster' when only the name is specified (e.g. 'etcd --name=abc')
	if cfg.Name != DefaultName && cfg.InitialCluster == defaultInitialCluster {
		cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	}

	cip, cport, _ := net.SplitHostPort(cfg.LCUrls[0].Host)
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
