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

package integration

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coreos/etcd/embed"
)

func TestEmbedEtcd(t *testing.T) {
	tests := []struct {
		cfg embed.Config

		werr     string
		wpeers   int
		wclients int
	}{
		{werr: "multiple discovery"},
		{werr: "advertise-client-urls is required"},
		{werr: "should be at least"},
		{werr: "is too long"},
		{wpeers: 1, wclients: 1},
		{wpeers: 2, wclients: 1},
		{wpeers: 1, wclients: 2},
	}

	urls := newEmbedURLs(10)

	// setup defaults
	for i := range tests {
		tests[i].cfg = *embed.NewConfig()
	}

	tests[0].cfg.Durl = "abc"
	setupEmbedCfg(&tests[1].cfg, []url.URL{urls[0]}, []url.URL{urls[1]})
	tests[1].cfg.ACUrls = nil
	tests[2].cfg.TickMs = tests[2].cfg.ElectionMs - 1
	tests[3].cfg.ElectionMs = 999999
	setupEmbedCfg(&tests[4].cfg, []url.URL{urls[2]}, []url.URL{urls[3]})
	setupEmbedCfg(&tests[5].cfg, []url.URL{urls[4]}, []url.URL{urls[5], urls[6]})
	setupEmbedCfg(&tests[6].cfg, []url.URL{urls[7], urls[8]}, []url.URL{urls[9]})

	dir := filepath.Join(os.TempDir(), fmt.Sprintf("embed-etcd"))
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	for i, tt := range tests {
		tests[i].cfg.Dir = dir
		e, err := embed.StartEtcd(&tests[i].cfg)
		if e != nil {
			<-e.Server.ReadyNotify() // wait for e.Server to join the cluster
		}
		if tt.werr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.werr) {
				t.Errorf("%d: expected error with %q, got %v", i, tt.werr, err)
			}
			if e != nil {
				e.Close()
			}
			continue
		}
		if err != nil {
			t.Errorf("%d: expected success, got error %v", i, err)
			continue
		}
		if len(e.Peers) != tt.wpeers {
			t.Errorf("%d: expected %d peers, got %d", i, tt.wpeers, len(e.Peers))
		}
		if len(e.Clients) != tt.wclients {
			t.Errorf("%d: expected %d peers, got %d", i, tt.wclients, len(e.Clients))
		}
		e.Close()
	}
}

func newEmbedURLs(n int) (urls []url.URL) {
	for i := 0; i < n; i++ {
		u, _ := url.Parse(fmt.Sprintf("unix://localhost:%d%06d", os.Getpid(), i))
		urls = append(urls, *u)
	}
	return
}

func setupEmbedCfg(cfg *embed.Config, curls []url.URL, purls []url.URL) {
	cfg.ClusterState = "new"
	cfg.LCUrls, cfg.ACUrls = curls, curls
	cfg.LPUrls, cfg.APUrls = purls, purls
	cfg.InitialCluster = ""
	for i := range purls {
		cfg.InitialCluster += ",default=" + purls[i].String()
	}
	cfg.InitialCluster = cfg.InitialCluster[1:]
}
