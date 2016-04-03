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

package etcdserver

import (
	"net/url"
	"testing"

	"github.com/coreos/etcd/pkg/types"
)

func mustNewURLs(t *testing.T, urls []string) []url.URL {
	if len(urls) == 0 {
		return nil
	}
	u, err := types.NewURLs(urls)
	if err != nil {
		t.Fatalf("error creating new URLs from %q: %v", urls, err)
	}
	return u
}

func TestConfigVerifyBootstrapWithoutClusterAndDiscoveryURLFail(t *testing.T) {
	c := &ServerConfig{
		Name:               "node1",
		DiscoveryURL:       "",
		InitialPeerURLsMap: types.URLsMap{},
	}
	if err := c.VerifyBootstrap(); err == nil {
		t.Errorf("err = nil, want not nil")
	}
}

func TestConfigVerifyExistingWithDiscoveryURLFail(t *testing.T) {
	cluster, err := types.NewURLsMap("node1=http://127.0.0.1:2380")
	if err != nil {
		t.Fatalf("NewCluster error: %v", err)
	}
	c := &ServerConfig{
		Name:               "node1",
		DiscoveryURL:       "http://127.0.0.1:2379/abcdefg",
		PeerURLs:           mustNewURLs(t, []string{"http://127.0.0.1:2380"}),
		InitialPeerURLsMap: cluster,
		NewCluster:         false,
	}
	if err := c.VerifyJoinExisting(); err == nil {
		t.Errorf("err = nil, want not nil")
	}
}

func TestConfigVerifyLocalMember(t *testing.T) {
	tests := []struct {
		clusterSetting string
		apurls         []string
		strict         bool
		shouldError    bool
	}{
		{
			// Node must exist in cluster
			"",
			nil,
			true,

			true,
		},
		{
			// Initial cluster set
			"node1=http://localhost:7001,node2=http://localhost:7002",
			[]string{"http://localhost:7001"},
			true,

			false,
		},
		{
			// Default initial cluster
			"node1=http://localhost:2380,node1=http://localhost:7001",
			[]string{"http://localhost:2380", "http://localhost:7001"},
			true,

			false,
		},
		{
			// Advertised peer URLs must match those in cluster-state
			"node1=http://localhost:7001",
			[]string{"http://localhost:12345"},
			true,

			true,
		},
		{
			// Advertised peer URLs must match those in cluster-state
			"node1=http://localhost:2380,node1=http://localhost:12345",
			[]string{"http://localhost:12345"},
			true,

			true,
		},
		{
			// Advertised peer URLs must match those in cluster-state
			"node1=http://localhost:2380",
			[]string{},
			true,

			true,
		},
		{
			// do not care about the urls if strict is not set
			"node1=http://localhost:2380",
			[]string{},
			false,

			false,
		},
	}

	for i, tt := range tests {
		cluster, err := types.NewURLsMap(tt.clusterSetting)
		if err != nil {
			t.Fatalf("#%d: Got unexpected error: %v", i, err)
		}
		cfg := ServerConfig{
			Name:               "node1",
			InitialPeerURLsMap: cluster,
		}
		if tt.apurls != nil {
			cfg.PeerURLs = mustNewURLs(t, tt.apurls)
		}
		err = cfg.verifyLocalMember(tt.strict)
		if (err == nil) && tt.shouldError {
			t.Errorf("#%d: Got no error where one was expected", i)
		}
		if (err != nil) && !tt.shouldError {
			t.Errorf("#%d: Got unexpected error: %v", i, err)
		}
	}
}

func TestSnapDir(t *testing.T) {
	tests := map[string]string{
		"/":            "/member/snap",
		"/var/lib/etc": "/var/lib/etc/member/snap",
	}
	for dd, w := range tests {
		cfg := ServerConfig{
			DataDir: dd,
		}
		if g := cfg.SnapDir(); g != w {
			t.Errorf("DataDir=%q: SnapDir()=%q, want=%q", dd, g, w)
		}
	}
}

func TestWALDir(t *testing.T) {
	tests := map[string]string{
		"/":            "/member/wal",
		"/var/lib/etc": "/var/lib/etc/member/wal",
	}
	for dd, w := range tests {
		cfg := ServerConfig{
			DataDir: dd,
		}
		if g := cfg.WALDir(); g != w {
			t.Errorf("DataDir=%q: WALDir()=%q, want=%q", dd, g, w)
		}
	}
}

func TestShouldDiscover(t *testing.T) {
	tests := map[string]bool{
		"":    false,
		"foo": true,
		"http://discovery.etcd.io/asdf": true,
	}
	for durl, w := range tests {
		cfg := ServerConfig{
			DiscoveryURL: durl,
		}
		if g := cfg.ShouldDiscover(); g != w {
			t.Errorf("durl=%q: ShouldDiscover()=%t, want=%t", durl, g, w)
		}
	}
}
