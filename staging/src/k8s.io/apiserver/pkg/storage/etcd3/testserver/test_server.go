/*
Copyright 2021 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/kubernetes"
	"go.etcd.io/etcd/server/v3/embed"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

// getAvailablePorts returns a desired count of TCP ports
// that are available for binding.
func getAvailablePorts(count int) ([]int, error) {
	ports := []int{}
	for i := 0; i < count; i++ {
		l, err := net.Listen("tcp", ":0")
		if err != nil {
			return nil, fmt.Errorf("could not bind to a port: %v", err)
		}
		// It is possible but unlikely that someone else will bind this port before we get a chance to use it.
		defer l.Close()
		ports = append(ports, l.Addr().(*net.TCPAddr).Port)
	}
	return ports, nil
}

// assignAvailablePorts assigns client and peer URLs to a cfg
func assignAvailablePorts(cfg *embed.Config) error {
	ports, err := getAvailablePorts(2)
	if err != nil {
		return err
	}

	// Only the port is replaced, so that a scheme or host the caller set before
	// starting the server survives a retry.
	setPort := func(urls []url.URL, port int) {
		for i := range urls {
			urls[i].Host = net.JoinHostPort(urls[i].Hostname(), strconv.Itoa(port))
		}
	}
	setPort(cfg.ListenClientUrls, ports[0])
	setPort(cfg.AdvertiseClientUrls, ports[0])
	setPort(cfg.ListenPeerUrls, ports[1])
	setPort(cfg.AdvertisePeerUrls, ports[1])
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	return nil
}

// NewTestConfig returns a configuration for an embedded etcd server.
// The configuration is based on embed.NewConfig(), with the following adjustments:
//   - sets UnsafeNoFsync = true to improve test performance (only reasonable in a test-only
//     single-member server we never intend to restart or keep data from)
//   - uses free ports for client and peer listeners
//   - cleans up the data directory on test termination
//   - silences server logs other than errors
func NewTestConfig(t testing.TB) *embed.Config {
	cfg := embed.NewConfig()

	cfg.UnsafeNoFsync = true

	if err := assignAvailablePorts(cfg); err != nil {
		t.Fatal(err)
	}

	cfg.ZapLoggerBuilder = embed.NewZapLoggerBuilder(zaptest.NewLogger(t, zaptest.Level(zapcore.ErrorLevel)).Named("etcd-server"))
	cfg.Dir = t.TempDir()
	os.Chmod(cfg.Dir, 0700)
	return cfg
}

// maxStartAttempts bounds how many times startEtcd retries with a fresh set of
// ports before giving up.
const maxStartAttempts = 3

var autoPortLock sync.Mutex

// startEtcd starts an embedded etcd server.
// Port assignment is subject to a race condition
// between assignment and binding, so we retry a few times.
func startEtcd(t testing.TB, cfg *embed.Config) (*embed.Config, *embed.Etcd, error) {
	t.Helper()

	autoPortLock.Lock()
	defer autoPortLock.Unlock()

	if cfg == nil {
		cfg = NewTestConfig(t)
	}

	for attempt := 1; ; attempt++ {
		e, err := embed.StartEtcd(cfg)
		if err == nil {
			return cfg, e, nil
		}
		if attempt >= maxStartAttempts || !errors.Is(err, syscall.EADDRINUSE) {
			return cfg, nil, err
		}
		t.Logf("etcd failed to bind on attempt %d of %d, retrying with new ports: %v", attempt, maxStartAttempts, err)
		if err := assignAvailablePorts(cfg); err != nil {
			return cfg, nil, err
		}
	}
}

// RunEtcd starts an embedded etcd server with the provided config
// (or NewTestConfig(t) if nil), and returns a client connected to the server.
// The server is terminated when the test ends.
func RunEtcd(t testing.TB, cfg *embed.Config) *kubernetes.Client {
	t.Helper()

	cfg, e, err := startEtcd(t, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)

	select {
	case <-e.Server.ReadyNotify():
	case <-time.After(60 * time.Second):
		e.Server.Stop() // trigger a shutdown
		t.Fatal("server took too long to start")
	}
	go func() {
		err := <-e.Err()
		if err != nil {
			t.Error(err)
		}
	}()

	tlsConfig, err := cfg.ClientTLSInfo.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}

	client, err := kubernetes.New(clientv3.Config{
		TLS:         tlsConfig,
		Endpoints:   e.Server.Cluster().ClientURLs(),
		DialTimeout: 10 * time.Second,
		Logger:      zaptest.NewLogger(t, zaptest.Level(zapcore.ErrorLevel)).Named("etcd-client"),
	})
	if err != nil {
		t.Fatal(err)
	}
	kubernetesRecorder := storagetesting.NewKubernetesRecorder(client.Kubernetes)
	client.KV = storagetesting.NewKVRecorder(client.KV, kubernetesRecorder)
	client.Kubernetes = kubernetesRecorder
	return client
}
