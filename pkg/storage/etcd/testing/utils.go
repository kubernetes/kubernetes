/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package testing

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/tools"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdhttp"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/rafthttp"
	goetcd "github.com/coreos/go-etcd/etcd"
)

// EtcdTestServer encapsulates the datastructures needed to start local instance for testing
type EtcdTestServer struct {
	etcdserver.ServerConfig
	PeerListeners, ClientListeners []net.Listener
	Client                         tools.EtcdClient

	raftHandler http.Handler
	s           *etcdserver.EtcdServer
	hss         []*httptest.Server
}

// newLocalListener opens a port localhost using any port
func newLocalListener(t *testing.T) net.Listener {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// configureTestCluster will set the params to start an etcd server
func configureTestCluster(t *testing.T, name string) *EtcdTestServer {
	var err error
	m := &EtcdTestServer{}

	pln := newLocalListener(t)
	m.PeerListeners = []net.Listener{pln}
	m.PeerURLs, err = types.NewURLs([]string{"http://" + pln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}

	cln := newLocalListener(t)
	m.ClientListeners = []net.Listener{cln}
	m.ClientURLs, err = types.NewURLs([]string{"http://" + cln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}

	m.Name = name
	m.DataDir, err = ioutil.TempDir(os.TempDir(), "etcd")
	if err != nil {
		t.Fatal(err)
	}

	clusterStr := fmt.Sprintf("%s=http://%s", name, pln.Addr().String())
	m.InitialPeerURLsMap, err = types.NewURLsMap(clusterStr)
	if err != nil {
		t.Fatal(err)
	}
	m.Transport, err = transport.NewTimeoutTransport(transport.TLSInfo{}, time.Second, rafthttp.ConnReadTimeout, rafthttp.ConnWriteTimeout)
	if err != nil {
		t.Fatal(err)
	}
	m.NewCluster = true
	m.ForceNewCluster = false
	m.ElectionTicks = 10
	m.TickMs = uint(10)

	return m
}

// launch will attempt to start the etcd server
func (m *EtcdTestServer) launch(t *testing.T) error {
	var err error
	if m.s, err = etcdserver.NewServer(&m.ServerConfig); err != nil {
		return fmt.Errorf("failed to initialize the etcd server: %v", err)
	}
	m.s.SyncTicker = time.Tick(500 * time.Millisecond)
	m.s.Start()
	m.raftHandler = etcdhttp.NewPeerHandler(m.s.Cluster(), m.s.RaftHandler())
	for _, ln := range m.PeerListeners {
		hs := &httptest.Server{
			Listener: ln,
			Config:   &http.Server{Handler: m.raftHandler},
		}
		hs.Start()
		m.hss = append(m.hss, hs)
	}
	for _, ln := range m.ClientListeners {
		hs := &httptest.Server{
			Listener: ln,
			Config:   &http.Server{Handler: etcdhttp.NewClientHandler(m.s, m.ServerConfig.ReqTimeout())},
		}
		hs.Start()
		m.hss = append(m.hss, hs)
	}
	return nil
}

// Terminate will shutdown the running etcd server
func (m *EtcdTestServer) Terminate(t *testing.T) {
	m.Client.(*goetcd.Client).Close()
	m.s.Stop()
	for _, hs := range m.hss {
		hs.CloseClientConnections()
		hs.Close()
	}
	if err := os.RemoveAll(m.ServerConfig.DataDir); err != nil {
		t.Fatal(err)
	}
}

// NewEtcdTestClientServer creates a new client and server for testing
func NewEtcdTestClientServer(t *testing.T) *EtcdTestServer {
	server := configureTestCluster(t, "foo")
	err := server.launch(t)
	if err != nil {
		t.Fatal("Failed to start etcd server error=%v", err)
		return nil
	}
	server.Client = goetcd.NewClient(server.ClientURLs.StringSlice())
	if server.Client == nil {
		t.Errorf("Failed to connect to local etcd server")
		defer server.Terminate(t)
		return nil
	}
	return server
}
