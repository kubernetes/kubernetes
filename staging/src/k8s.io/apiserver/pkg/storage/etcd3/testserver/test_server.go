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
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/kubernetes"
	"go.etcd.io/etcd/server/v3/embed"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

// getAvailablePort returns a TCP port that is available for binding.
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

	ports, err := getAvailablePorts(2)
	if err != nil {
		t.Fatal(err)
	}
	clientURL := url.URL{Scheme: "http", Host: net.JoinHostPort("localhost", strconv.Itoa(ports[0]))}
	peerURL := url.URL{Scheme: "http", Host: net.JoinHostPort("localhost", strconv.Itoa(ports[1]))}

	cfg.ListenPeerUrls = []url.URL{peerURL}
	cfg.AdvertisePeerUrls = []url.URL{peerURL}
	cfg.ListenClientUrls = []url.URL{clientURL}
	cfg.AdvertiseClientUrls = []url.URL{clientURL}
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)

	cfg.ZapLoggerBuilder = embed.NewZapLoggerBuilder(zaptest.NewLogger(t, zaptest.Level(zapcore.ErrorLevel)).Named("etcd-server"))
	cfg.Dir = t.TempDir()
	os.Chmod(cfg.Dir, 0700)
	return cfg
}

var autoPortLock sync.Mutex

// RunEtcd starts an embedded etcd server with the provided config
// (or NewTestConfig(t) if nil), and returns a client connected to the server.
// The server is terminated when the test ends.
func RunEtcd(t testing.TB, cfg *embed.Config) *kubernetes.Client {
	t.Helper()

	if cfg == nil {
		// if we have to autopick free ports, lock until we successfully start the server on the ports we chose
		autoPortLock.Lock()
		defer autoPortLock.Unlock()
		cfg = NewTestConfig(t)
	}

	e, err := embed.StartEtcd(cfg)
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

	return newTestClient(t, clientv3.Config{
		TLS:       tlsConfig,
		Endpoints: e.Server.Cluster().ClientURLs(),
	})
}

// RunExternalEtcd starts an external etcd subprocess and returns a client connected to the server.
// The external etcd process is terminated when the test ends.
func RunExternalEtcd(t testing.TB) *kubernetes.Client {
	t.Helper()
	ports, err := getAvailablePorts(2)
	if err != nil {
		t.Fatal(err)
	}
	clientPort := ports[0]
	peerPort := ports[1]

	clientURL := fmt.Sprintf("http://127.0.0.1:%d", clientPort)
	peerURL := fmt.Sprintf("http://127.0.0.1:%d", peerPort)

	dir := t.TempDir()

	cmd := exec.Command("etcd",
		"--data-dir", dir,
		"--listen-client-urls", clientURL,
		"--advertise-client-urls", clientURL,
		"--listen-peer-urls", peerURL,
		"--initial-advertise-peer-urls", peerURL,
		"--initial-cluster", fmt.Sprintf("default=%s", peerURL),
		"--quota-backend-bytes", strconv.FormatInt(8*1024*1024*1024, 10),
		"--log-level", "warn",
	)

	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start external etcd: %v", err)
	}

	ready := false
	for i := 0; i < 50; i++ {
		conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(clientPort)), 100*time.Millisecond)
		if err == nil {
			conn.Close()
			ready = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !ready {
		cmd.Process.Kill()
		t.Fatal("external etcd failed to start")
	}

	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})

	return newTestClient(t, clientv3.Config{
		Endpoints: []string{clientURL},
	})
}

func newTestClient(t testing.TB, config clientv3.Config) *kubernetes.Client {
	if config.Logger == nil {
		config.Logger = zaptest.NewLogger(t, zaptest.Level(zapcore.ErrorLevel)).Named("etcd-client")
	}
	if config.DialTimeout == 0 {
		config.DialTimeout = 10 * time.Second
	}
	client, err := kubernetes.New(config)
	if err != nil {
		t.Fatal(err)
	}
	kubernetesRecorder := storagetesting.NewKubernetesRecorder(client.Kubernetes)
	client.KV = storagetesting.NewKVRecorder(client.KV, kubernetesRecorder)
	client.Kubernetes = kubernetesRecorder
	return client
}
