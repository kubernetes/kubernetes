/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testserver

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	"go.etcd.io/etcd/server/v3/embed"
)

// TestRunEtcdRecoversFromTakenPort claims a port that was selected for etcd
// before it gets a chance to bind, reproducing what happens when another
// process takes the port first, and verifies that RunEtcd recovers instead of
// failing the test. Both listeners are covered because etcd binds the peer
// listener before the client one, so either can be the one that loses the race.
func TestRunEtcdRecoversFromTakenPort(t *testing.T) {
	for _, tc := range []struct {
		name string
		addr func(cfg *embed.Config) string
	}{
		{name: "client", addr: func(cfg *embed.Config) string { return cfg.ListenClientUrls[0].Host }},
		{name: "peer", addr: func(cfg *embed.Config) string { return cfg.ListenPeerUrls[0].Host }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := NewTestConfig(t)
			taken := tc.addr(cfg)

			l, err := net.Listen("tcp", taken)
			if err != nil {
				t.Fatalf("failed to claim %s: %v", taken, err)
			}
			defer func() { _ = l.Close() }()

			client := RunEtcd(t, cfg)

			if got := tc.addr(cfg); got == taken {
				t.Errorf("etcd is still configured to listen on the claimed address %s", taken)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if _, err := client.KV.Get(ctx, "/registry"); err != nil {
				t.Fatalf("etcd did not serve requests after moving to a new port: %v", err)
			}
		})
	}
}

// TestAssignAvailablePortsPreservesURLs verifies that reassigning ports leaves
// the rest of the URL alone. Callers such as the storagebackend TLS test switch
// the scheme of the URLs returned by NewTestConfig to https, and that has to
// survive a retry.
func TestAssignAvailablePortsPreservesURLs(t *testing.T) {
	cfg := NewTestConfig(t)
	for i := range cfg.ListenClientUrls {
		cfg.ListenClientUrls[i].Scheme = "https"
	}
	clientPort := cfg.ListenClientUrls[0].Port()
	peerPort := cfg.ListenPeerUrls[0].Port()

	if err := assignAvailablePorts(cfg); err != nil {
		t.Fatal(err)
	}

	if got, want := cfg.ListenClientUrls[0].Scheme, "https"; got != want {
		t.Errorf("client URL scheme = %q, want %q", got, want)
	}
	if got, want := cfg.ListenClientUrls[0].Hostname(), "localhost"; got != want {
		t.Errorf("client URL hostname = %q, want %q", got, want)
	}
	if got := cfg.ListenClientUrls[0].Port(); got == clientPort {
		t.Errorf("client port was not reassigned, still %q", got)
	}
	if got := cfg.ListenPeerUrls[0].Port(); got == peerPort {
		t.Errorf("peer port was not reassigned, still %q", got)
	}
	if cfg.ListenClientUrls[0].Port() == cfg.ListenPeerUrls[0].Port() {
		t.Error("client and peer listeners must not be assigned the same port")
	}
	if got, want := cfg.InitialCluster, cfg.AdvertisePeerUrls[0].String(); !strings.Contains(got, want) {
		t.Errorf("InitialCluster = %q, want it to reference the reassigned peer URL %q", got, want)
	}
}
