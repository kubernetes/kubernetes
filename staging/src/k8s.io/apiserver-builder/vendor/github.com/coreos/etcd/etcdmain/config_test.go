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
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ghodss/yaml"
)

func TestConfigParsingMemberFlags(t *testing.T) {
	args := []string{
		"-data-dir=testdir",
		"-name=testname",
		"-max-wals=10",
		"-max-snapshots=10",
		"-snapshot-count=10",
		"-listen-peer-urls=http://localhost:8000,https://localhost:8001",
		"-listen-client-urls=http://localhost:7000,https://localhost:7001",
		// it should be set if -listen-client-urls is set
		"-advertise-client-urls=http://localhost:7000,https://localhost:7001",
	}

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateMemberFlags(t, cfg)
}

func TestConfigFileMemberFields(t *testing.T) {
	yc := struct {
		Dir           string `json:"data-dir"`
		MaxSnapFiles  uint   `json:"max-snapshots"`
		MaxWalFiles   uint   `json:"max-wals"`
		Name          string `json:"name"`
		SnapCount     uint64 `json:"snapshot-count"`
		LPUrls        string `json:"listen-peer-urls"`
		LCUrls        string `json:"listen-client-urls"`
		AcurlsCfgFile string `json:"advertise-client-urls"`
	}{
		"testdir",
		10,
		10,
		"testname",
		10,
		"http://localhost:8000,https://localhost:8001",
		"http://localhost:7000,https://localhost:7001",
		"http://localhost:7000,https://localhost:7001",
	}

	b, err := yaml.Marshal(&yc)
	if err != nil {
		t.Fatal(err)
	}

	tmpfile := mustCreateCfgFile(t, b)
	defer os.Remove(tmpfile.Name())

	args := []string{
		fmt.Sprintf("--config-file=%s", tmpfile.Name()),
	}

	cfg := NewConfig()
	err = cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateMemberFlags(t, cfg)
}

func TestConfigParsingClusteringFlags(t *testing.T) {
	args := []string{
		"-initial-cluster=0=http://localhost:8000",
		"-initial-cluster-state=existing",
		"-initial-cluster-token=etcdtest",
		"-initial-advertise-peer-urls=http://localhost:8000,https://localhost:8001",
		"-advertise-client-urls=http://localhost:7000,https://localhost:7001",
		"-discovery-fallback=exit",
	}

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateClusteringFlags(t, cfg)
}

func TestConfigFileClusteringFields(t *testing.T) {
	yc := struct {
		InitialCluster      string `json:"initial-cluster"`
		ClusterState        string `json:"initial-cluster-state"`
		InitialClusterToken string `json:"initial-cluster-token"`
		Apurls              string `json:"initial-advertise-peer-urls"`
		Acurls              string `json:"advertise-client-urls"`
		Fallback            string `json:"discovery-fallback"`
	}{
		"0=http://localhost:8000",
		"existing",
		"etcdtest",
		"http://localhost:8000,https://localhost:8001",
		"http://localhost:7000,https://localhost:7001",
		"exit",
	}

	b, err := yaml.Marshal(&yc)
	if err != nil {
		t.Fatal(err)
	}

	tmpfile := mustCreateCfgFile(t, b)
	defer os.Remove(tmpfile.Name())

	args := []string{
		fmt.Sprintf("--config-file=%s", tmpfile.Name()),
	}
	cfg := NewConfig()
	err = cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateClusteringFlags(t, cfg)
}

func TestConfigParsingOtherFlags(t *testing.T) {
	args := []string{
		"-proxy=readonly",
		"-ca-file=cafile",
		"-cert-file=certfile",
		"-key-file=keyfile",
		"-peer-ca-file=peercafile",
		"-peer-cert-file=peercertfile",
		"-peer-key-file=peerkeyfile",
		"-force-new-cluster=true",
	}

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateOtherFlags(t, cfg)
}

func TestConfigFileOtherFields(t *testing.T) {
	yc := struct {
		ProxyCfgFile          string         `json:"proxy"`
		ClientSecurityCfgFile securityConfig `json:"client-transport-security"`
		PeerSecurityCfgFile   securityConfig `json:"peer-transport-security"`
		ForceNewCluster       bool           `json:"force-new-cluster"`
	}{
		"readonly",
		securityConfig{
			CAFile:   "cafile",
			CertFile: "certfile",
			KeyFile:  "keyfile",
		},
		securityConfig{
			CAFile:   "peercafile",
			CertFile: "peercertfile",
			KeyFile:  "peerkeyfile",
		},
		true,
	}

	b, err := yaml.Marshal(&yc)
	if err != nil {
		t.Fatal(err)
	}

	tmpfile := mustCreateCfgFile(t, b)
	defer os.Remove(tmpfile.Name())

	args := []string{
		fmt.Sprintf("--config-file=%s", tmpfile.Name()),
	}

	cfg := NewConfig()
	err = cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}

	validateOtherFlags(t, cfg)
}

func TestConfigParsingConflictClusteringFlags(t *testing.T) {
	conflictArgs := [][]string{
		{
			"-initial-cluster=0=localhost:8000",
			"-discovery=http://example.com/abc",
		},
		{
			"-discovery-srv=example.com",
			"-discovery=http://example.com/abc",
		},
		{
			"-initial-cluster=0=localhost:8000",
			"-discovery-srv=example.com",
		},
		{
			"-initial-cluster=0=localhost:8000",
			"-discovery=http://example.com/abc",
			"-discovery-srv=example.com",
		},
	}

	for i, tt := range conflictArgs {
		cfg := NewConfig()
		err := cfg.Parse(tt)
		if err != ErrConflictBootstrapFlags {
			t.Errorf("%d: err = %v, want %v", i, err, ErrConflictBootstrapFlags)
		}
	}
}

func TestConfigFileConflictClusteringFlags(t *testing.T) {
	tests := []struct {
		InitialCluster string `json:"initial-cluster"`
		DnsCluster     string `json:"discovery-srv"`
		Durl           string `json:"discovery"`
	}{
		{
			InitialCluster: "0=localhost:8000",
			Durl:           "http://example.com/abc",
		},
		{
			DnsCluster: "example.com",
			Durl:       "http://example.com/abc",
		},
		{
			InitialCluster: "0=localhost:8000",
			DnsCluster:     "example.com",
		},
		{
			InitialCluster: "0=localhost:8000",
			Durl:           "http://example.com/abc",
			DnsCluster:     "example.com",
		},
	}

	for i, tt := range tests {
		b, err := yaml.Marshal(&tt)
		if err != nil {
			t.Fatal(err)
		}

		tmpfile := mustCreateCfgFile(t, b)
		defer os.Remove(tmpfile.Name())

		args := []string{
			fmt.Sprintf("--config-file=%s", tmpfile.Name()),
		}

		cfg := NewConfig()
		err = cfg.Parse(args)
		if err != ErrConflictBootstrapFlags {
			t.Errorf("%d: err = %v, want %v", i, err, ErrConflictBootstrapFlags)
		}
	}
}

func TestConfigParsingMissedAdvertiseClientURLsFlag(t *testing.T) {
	tests := []struct {
		args []string
		werr error
	}{
		{
			[]string{
				"-initial-cluster=infra1=http://127.0.0.1:2380",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			errUnsetAdvertiseClientURLsFlag,
		},
		{
			[]string{
				"-discovery-srv=example.com",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			errUnsetAdvertiseClientURLsFlag,
		},
		{
			[]string{
				"-discovery=http://example.com/abc",
				"-discovery-fallback=exit",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			errUnsetAdvertiseClientURLsFlag,
		},
		{
			[]string{
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			errUnsetAdvertiseClientURLsFlag,
		},
		{
			[]string{
				"-discovery=http://example.com/abc",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			nil,
		},
		{
			[]string{
				"-proxy=on",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			nil,
		},
		{
			[]string{
				"-proxy=readonly",
				"-listen-client-urls=http://127.0.0.1:2379",
			},
			nil,
		},
	}

	for i, tt := range tests {
		cfg := NewConfig()
		err := cfg.Parse(tt.args)
		if err != tt.werr {
			t.Errorf("%d: err = %v, want %v", i, err, tt.werr)
		}
	}
}

func TestConfigIsNewCluster(t *testing.T) {
	tests := []struct {
		state  string
		wIsNew bool
	}{
		{clusterStateFlagExisting, false},
		{clusterStateFlagNew, true},
	}
	for i, tt := range tests {
		cfg := NewConfig()
		if err := cfg.clusterState.Set(tt.state); err != nil {
			t.Fatalf("#%d: unexpected clusterState.Set error: %v", i, err)
		}
		if g := cfg.isNewCluster(); g != tt.wIsNew {
			t.Errorf("#%d: isNewCluster = %v, want %v", i, g, tt.wIsNew)
		}
	}
}

func TestConfigIsProxy(t *testing.T) {
	tests := []struct {
		proxy    string
		wIsProxy bool
	}{
		{proxyFlagOff, false},
		{proxyFlagReadonly, true},
		{proxyFlagOn, true},
	}
	for i, tt := range tests {
		cfg := NewConfig()
		if err := cfg.proxy.Set(tt.proxy); err != nil {
			t.Fatalf("#%d: unexpected proxy.Set error: %v", i, err)
		}
		if g := cfg.isProxy(); g != tt.wIsProxy {
			t.Errorf("#%d: isProxy = %v, want %v", i, g, tt.wIsProxy)
		}
	}
}

func TestConfigIsReadonlyProxy(t *testing.T) {
	tests := []struct {
		proxy       string
		wIsReadonly bool
	}{
		{proxyFlagOff, false},
		{proxyFlagReadonly, true},
		{proxyFlagOn, false},
	}
	for i, tt := range tests {
		cfg := NewConfig()
		if err := cfg.proxy.Set(tt.proxy); err != nil {
			t.Fatalf("#%d: unexpected proxy.Set error: %v", i, err)
		}
		if g := cfg.isReadonlyProxy(); g != tt.wIsReadonly {
			t.Errorf("#%d: isReadonlyProxy = %v, want %v", i, g, tt.wIsReadonly)
		}
	}
}

func TestConfigShouldFallbackToProxy(t *testing.T) {
	tests := []struct {
		fallback  string
		wFallback bool
	}{
		{fallbackFlagProxy, true},
		{fallbackFlagExit, false},
	}
	for i, tt := range tests {
		cfg := NewConfig()
		if err := cfg.fallback.Set(tt.fallback); err != nil {
			t.Fatalf("#%d: unexpected fallback.Set error: %v", i, err)
		}
		if g := cfg.shouldFallbackToProxy(); g != tt.wFallback {
			t.Errorf("#%d: shouldFallbackToProxy = %v, want %v", i, g, tt.wFallback)
		}
	}
}

func TestConfigFileElectionTimeout(t *testing.T) {
	tests := []struct {
		TickMs     uint `json:"heartbeat-interval"`
		ElectionMs uint `json:"election-timeout"`
		errStr     string
	}{
		{
			ElectionMs: 1000,
			TickMs:     800,
			errStr:     "should be at least as 5 times as",
		},
		{
			ElectionMs: 60000,
			errStr:     "is too long, and should be set less than",
		},
	}

	for i, tt := range tests {
		b, err := yaml.Marshal(&tt)
		if err != nil {
			t.Fatal(err)
		}

		tmpfile := mustCreateCfgFile(t, b)
		defer os.Remove(tmpfile.Name())

		args := []string{
			fmt.Sprintf("--config-file=%s", tmpfile.Name()),
		}

		cfg := NewConfig()
		err = cfg.Parse(args)
		if !strings.Contains(err.Error(), tt.errStr) {
			t.Errorf("%d: Wrong err = %v", i, err)
		}
	}
}

func mustCreateCfgFile(t *testing.T, b []byte) *os.File {
	tmpfile, err := ioutil.TempFile("", "servercfg")
	if err != nil {
		t.Fatal(err)
	}

	_, err = tmpfile.Write(b)
	if err != nil {
		t.Fatal(err)
	}
	err = tmpfile.Close()
	if err != nil {
		t.Fatal(err)
	}

	return tmpfile
}

func validateMemberFlags(t *testing.T, cfg *config) {
	wcfg := &config{
		Dir:          "testdir",
		lpurls:       []url.URL{{Scheme: "http", Host: "localhost:8000"}, {Scheme: "https", Host: "localhost:8001"}},
		lcurls:       []url.URL{{Scheme: "http", Host: "localhost:7000"}, {Scheme: "https", Host: "localhost:7001"}},
		MaxSnapFiles: 10,
		MaxWalFiles:  10,
		Name:         "testname",
		SnapCount:    10,
	}

	if cfg.Dir != wcfg.Dir {
		t.Errorf("dir = %v, want %v", cfg.Dir, wcfg.Dir)
	}
	if cfg.MaxSnapFiles != wcfg.MaxSnapFiles {
		t.Errorf("maxsnap = %v, want %v", cfg.MaxSnapFiles, wcfg.MaxSnapFiles)
	}
	if cfg.MaxWalFiles != wcfg.MaxWalFiles {
		t.Errorf("maxwal = %v, want %v", cfg.MaxWalFiles, wcfg.MaxWalFiles)
	}
	if cfg.Name != wcfg.Name {
		t.Errorf("name = %v, want %v", cfg.Name, wcfg.Name)
	}
	if cfg.SnapCount != wcfg.SnapCount {
		t.Errorf("snapcount = %v, want %v", cfg.SnapCount, wcfg.SnapCount)
	}
	if !reflect.DeepEqual(cfg.lpurls, wcfg.lpurls) {
		t.Errorf("listen-peer-urls = %v, want %v", cfg.lpurls, wcfg.lpurls)
	}
	if !reflect.DeepEqual(cfg.lcurls, wcfg.lcurls) {
		t.Errorf("listen-client-urls = %v, want %v", cfg.lcurls, wcfg.lcurls)
	}
}

func validateClusteringFlags(t *testing.T, cfg *config) {
	wcfg := NewConfig()
	wcfg.apurls = []url.URL{{Scheme: "http", Host: "localhost:8000"}, {Scheme: "https", Host: "localhost:8001"}}
	wcfg.acurls = []url.URL{{Scheme: "http", Host: "localhost:7000"}, {Scheme: "https", Host: "localhost:7001"}}
	wcfg.clusterState.Set(clusterStateFlagExisting)
	wcfg.fallback.Set(fallbackFlagExit)
	wcfg.InitialCluster = "0=http://localhost:8000"
	wcfg.InitialClusterToken = "etcdtest"

	if cfg.clusterState.String() != wcfg.clusterState.String() {
		t.Errorf("clusterState = %v, want %v", cfg.clusterState, wcfg.clusterState)
	}
	if cfg.fallback.String() != wcfg.fallback.String() {
		t.Errorf("fallback = %v, want %v", cfg.fallback, wcfg.fallback)
	}
	if cfg.InitialCluster != wcfg.InitialCluster {
		t.Errorf("initialCluster = %v, want %v", cfg.InitialCluster, wcfg.InitialCluster)
	}
	if cfg.InitialClusterToken != wcfg.InitialClusterToken {
		t.Errorf("initialClusterToken = %v, want %v", cfg.InitialClusterToken, wcfg.InitialClusterToken)
	}
	if !reflect.DeepEqual(cfg.apurls, wcfg.apurls) {
		t.Errorf("initial-advertise-peer-urls = %v, want %v", cfg.lpurls, wcfg.lpurls)
	}
	if !reflect.DeepEqual(cfg.acurls, wcfg.acurls) {
		t.Errorf("advertise-client-urls = %v, want %v", cfg.lcurls, wcfg.lcurls)
	}
}

func validateOtherFlags(t *testing.T, cfg *config) {
	wcfg := NewConfig()
	wcfg.proxy.Set(proxyFlagReadonly)
	wcfg.clientTLSInfo.CAFile = "cafile"
	wcfg.clientTLSInfo.CertFile = "certfile"
	wcfg.clientTLSInfo.KeyFile = "keyfile"
	wcfg.peerTLSInfo.CAFile = "peercafile"
	wcfg.peerTLSInfo.CertFile = "peercertfile"
	wcfg.peerTLSInfo.KeyFile = "peerkeyfile"
	wcfg.ForceNewCluster = true

	if cfg.proxy.String() != wcfg.proxy.String() {
		t.Errorf("proxy = %v, want %v", cfg.proxy, wcfg.proxy)
	}
	if cfg.clientTLSInfo.String() != wcfg.clientTLSInfo.String() {
		t.Errorf("clientTLS = %v, want %v", cfg.clientTLSInfo, wcfg.clientTLSInfo)
	}
	if cfg.peerTLSInfo.String() != wcfg.peerTLSInfo.String() {
		t.Errorf("peerTLS = %v, want %v", cfg.peerTLSInfo, wcfg.peerTLSInfo)
	}
	if cfg.ForceNewCluster != wcfg.ForceNewCluster {
		t.Errorf("forceNewCluster = %t, want %t", cfg.ForceNewCluster, wcfg.ForceNewCluster)
	}
}
