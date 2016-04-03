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

package etcdmain

import (
	"net/url"
	"reflect"
	"testing"
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
	wcfg := &config{
		dir:          "testdir",
		lpurls:       []url.URL{{Scheme: "http", Host: "localhost:8000"}, {Scheme: "https", Host: "localhost:8001"}},
		lcurls:       []url.URL{{Scheme: "http", Host: "localhost:7000"}, {Scheme: "https", Host: "localhost:7001"}},
		maxSnapFiles: 10,
		maxWalFiles:  10,
		name:         "testname",
		snapCount:    10,
	}

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.dir != wcfg.dir {
		t.Errorf("dir = %v, want %v", cfg.dir, wcfg.dir)
	}
	if cfg.maxSnapFiles != wcfg.maxSnapFiles {
		t.Errorf("maxsnap = %v, want %v", cfg.maxSnapFiles, wcfg.maxSnapFiles)
	}
	if cfg.maxWalFiles != wcfg.maxWalFiles {
		t.Errorf("maxwal = %v, want %v", cfg.maxWalFiles, wcfg.maxWalFiles)
	}
	if cfg.name != wcfg.name {
		t.Errorf("name = %v, want %v", cfg.name, wcfg.name)
	}
	if cfg.snapCount != wcfg.snapCount {
		t.Errorf("snapcount = %v, want %v", cfg.snapCount, wcfg.snapCount)
	}
	if !reflect.DeepEqual(cfg.lpurls, wcfg.lpurls) {
		t.Errorf("listen-peer-urls = %v, want %v", cfg.lpurls, wcfg.lpurls)
	}
	if !reflect.DeepEqual(cfg.lcurls, wcfg.lcurls) {
		t.Errorf("listen-client-urls = %v, want %v", cfg.lcurls, wcfg.lcurls)
	}
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
	wcfg := NewConfig()
	wcfg.apurls = []url.URL{{Scheme: "http", Host: "localhost:8000"}, {Scheme: "https", Host: "localhost:8001"}}
	wcfg.acurls = []url.URL{{Scheme: "http", Host: "localhost:7000"}, {Scheme: "https", Host: "localhost:7001"}}
	wcfg.clusterState.Set(clusterStateFlagExisting)
	wcfg.fallback.Set(fallbackFlagExit)
	wcfg.initialCluster = "0=http://localhost:8000"
	wcfg.initialClusterToken = "etcdtest"

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.clusterState.String() != wcfg.clusterState.String() {
		t.Errorf("clusterState = %v, want %v", cfg.clusterState, wcfg.clusterState)
	}
	if cfg.fallback.String() != wcfg.fallback.String() {
		t.Errorf("fallback = %v, want %v", cfg.fallback, wcfg.fallback)
	}
	if cfg.initialCluster != wcfg.initialCluster {
		t.Errorf("initialCluster = %v, want %v", cfg.initialCluster, wcfg.initialCluster)
	}
	if cfg.initialClusterToken != wcfg.initialClusterToken {
		t.Errorf("initialClusterToken = %v, want %v", cfg.initialClusterToken, wcfg.initialClusterToken)
	}
	if !reflect.DeepEqual(cfg.apurls, wcfg.apurls) {
		t.Errorf("initial-advertise-peer-urls = %v, want %v", cfg.lpurls, wcfg.lpurls)
	}
	if !reflect.DeepEqual(cfg.acurls, wcfg.acurls) {
		t.Errorf("advertise-client-urls = %v, want %v", cfg.lcurls, wcfg.lcurls)
	}
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

	wcfg := NewConfig()
	wcfg.proxy.Set(proxyFlagReadonly)
	wcfg.clientTLSInfo.CAFile = "cafile"
	wcfg.clientTLSInfo.CertFile = "certfile"
	wcfg.clientTLSInfo.KeyFile = "keyfile"
	wcfg.peerTLSInfo.CAFile = "peercafile"
	wcfg.peerTLSInfo.CertFile = "peercertfile"
	wcfg.peerTLSInfo.KeyFile = "peerkeyfile"
	wcfg.forceNewCluster = true

	cfg := NewConfig()
	err := cfg.Parse(args)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.proxy.String() != wcfg.proxy.String() {
		t.Errorf("proxy = %v, want %v", cfg.proxy, wcfg.proxy)
	}
	if cfg.clientTLSInfo.String() != wcfg.clientTLSInfo.String() {
		t.Errorf("clientTLS = %v, want %v", cfg.clientTLSInfo, wcfg.clientTLSInfo)
	}
	if cfg.peerTLSInfo.String() != wcfg.peerTLSInfo.String() {
		t.Errorf("peerTLS = %v, want %v", cfg.peerTLSInfo, wcfg.peerTLSInfo)
	}
	if cfg.forceNewCluster != wcfg.forceNewCluster {
		t.Errorf("forceNewCluster = %t, want %t", cfg.forceNewCluster, wcfg.forceNewCluster)
	}
}

func TestConfigParsingV1Flags(t *testing.T) {
	args := []string{
		"-peer-addr=127.0.0.1:2380",
		"-addr=127.0.0.1:2379",
	}
	wcfg := NewConfig()
	wcfg.lpurls = []url.URL{{Scheme: "http", Host: "[::]:2380"}}
	wcfg.apurls = []url.URL{{Scheme: "http", Host: "127.0.0.1:2380"}}
	wcfg.lcurls = []url.URL{{Scheme: "http", Host: "[::]:2379"}}
	wcfg.acurls = []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}}

	cfg := NewConfig()
	if err := cfg.Parse(args); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cfg.lpurls, wcfg.lpurls) {
		t.Errorf("listen peer urls = %+v, want %+v", cfg.lpurls, wcfg.lpurls)
	}
	if !reflect.DeepEqual(cfg.apurls, wcfg.apurls) {
		t.Errorf("advertise peer urls = %+v, want %+v", cfg.apurls, wcfg.apurls)
	}
	if !reflect.DeepEqual(cfg.lcurls, wcfg.lcurls) {
		t.Errorf("listen client urls = %+v, want %+v", cfg.lcurls, wcfg.lcurls)
	}
	if !reflect.DeepEqual(cfg.acurls, wcfg.acurls) {
		t.Errorf("advertise client urls = %+v, want %+v", cfg.acurls, wcfg.acurls)
	}
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
