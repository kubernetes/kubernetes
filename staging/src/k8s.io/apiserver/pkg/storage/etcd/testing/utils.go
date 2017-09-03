/*
Copyright 2015 The Kubernetes Authors.

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
	"path"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/etcd/etcdtest"
	"k8s.io/apiserver/pkg/storage/etcd/testing/testingcert"
	"k8s.io/apiserver/pkg/storage/storagebackend"

	etcd "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// EtcdTestServer encapsulates the datastructures needed to start local instance for testing
type EtcdTestServer struct {
	// The following are lumped etcd2 test server params
	// TODO: Deprecate in a post 1.5 release
	etcdserver.ServerConfig
	PeerListeners, ClientListeners []net.Listener
	Client                         etcd.Client

	CertificatesDir string
	CertFile        string
	KeyFile         string
	CAFile          string

	raftHandler http.Handler
	s           *etcdserver.EtcdServer
	hss         []*httptest.Server

	// The following are lumped etcd3 test server params
	v3Cluster *integration.ClusterV3
	V3Client  *clientv3.Client
}

// newLocalListener opens a port localhost using any port
func newLocalListener(t *testing.T) net.Listener {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// newSecuredLocalListener opens a port localhost using any port
// with SSL enable
func newSecuredLocalListener(t *testing.T, certFile, keyFile, caFile string) net.Listener {
	var l net.Listener
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	tlsInfo := transport.TLSInfo{
		CertFile: certFile,
		KeyFile:  keyFile,
		CAFile:   caFile,
	}
	tlscfg, err := tlsInfo.ServerConfig()
	if err != nil {
		t.Fatalf("unexpected serverConfig error: %v", err)
	}
	l, err = transport.NewKeepAliveListener(l, "https", tlscfg)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func newHttpTransport(t *testing.T, certFile, keyFile, caFile string) etcd.CancelableTransport {
	tlsInfo := transport.TLSInfo{
		CertFile: certFile,
		KeyFile:  keyFile,
		CAFile:   caFile,
	}
	tr, err := transport.NewTransport(tlsInfo, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

// configureTestCluster will set the params to start an etcd server
func configureTestCluster(t *testing.T, name string, https bool) *EtcdTestServer {
	var err error
	m := &EtcdTestServer{}

	pln := newLocalListener(t)
	m.PeerListeners = []net.Listener{pln}
	m.PeerURLs, err = types.NewURLs([]string{"http://" + pln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}

	// Allow test launches to control where etcd data goes, for space or performance reasons
	baseDir := os.Getenv("TEST_ETCD_DIR")
	if len(baseDir) == 0 {
		baseDir = os.TempDir()
	}

	if https {
		m.CertificatesDir, err = ioutil.TempDir(baseDir, "etcd_certificates")
		if err != nil {
			t.Fatal(err)
		}
		m.CertFile = path.Join(m.CertificatesDir, "etcdcert.pem")
		if err = ioutil.WriteFile(m.CertFile, []byte(testingcert.CertFileContent), 0644); err != nil {
			t.Fatal(err)
		}
		m.KeyFile = path.Join(m.CertificatesDir, "etcdkey.pem")
		if err = ioutil.WriteFile(m.KeyFile, []byte(testingcert.KeyFileContent), 0644); err != nil {
			t.Fatal(err)
		}
		m.CAFile = path.Join(m.CertificatesDir, "ca.pem")
		if err = ioutil.WriteFile(m.CAFile, []byte(testingcert.CAFileContent), 0644); err != nil {
			t.Fatal(err)
		}

		cln := newSecuredLocalListener(t, m.CertFile, m.KeyFile, m.CAFile)
		m.ClientListeners = []net.Listener{cln}
		m.ClientURLs, err = types.NewURLs([]string{"https://" + cln.Addr().String()})
		if err != nil {
			t.Fatal(err)
		}
	} else {
		cln := newLocalListener(t)
		m.ClientListeners = []net.Listener{cln}
		m.ClientURLs, err = types.NewURLs([]string{"http://" + cln.Addr().String()})
		if err != nil {
			t.Fatal(err)
		}
	}

	m.Name = name
	m.DataDir, err = ioutil.TempDir(baseDir, "etcd")
	if err != nil {
		t.Fatal(err)
	}

	clusterStr := fmt.Sprintf("%s=http://%s", name, pln.Addr().String())
	m.InitialPeerURLsMap, err = types.NewURLsMap(clusterStr)
	if err != nil {
		t.Fatal(err)
	}
	m.InitialClusterToken = "TestEtcd"
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
	m.raftHandler = &testutil.PauseableHandler{Next: v2http.NewPeerHandler(m.s)}
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
			Config:   &http.Server{Handler: v2http.NewClientHandler(m.s, m.ServerConfig.ReqTimeout())},
		}
		hs.Start()
		m.hss = append(m.hss, hs)
	}
	return nil
}

// waitForEtcd wait until etcd is propagated correctly
func (m *EtcdTestServer) waitUntilUp() error {
	membersAPI := etcd.NewMembersAPI(m.Client)
	for start := time.Now(); time.Since(start) < wait.ForeverTestTimeout; time.Sleep(10 * time.Millisecond) {
		members, err := membersAPI.List(context.TODO())
		if err != nil {
			glog.Errorf("Error when getting etcd cluster members")
			continue
		}
		if len(members) == 1 && len(members[0].ClientURLs) > 0 {
			return nil
		}
	}
	return fmt.Errorf("timeout on waiting for etcd cluster")
}

// Terminate will shutdown the running etcd server
func (m *EtcdTestServer) Terminate(t *testing.T) {
	if m.v3Cluster != nil {
		m.v3Cluster.Terminate(t)
	} else {
		m.Client = nil
		m.s.Stop()
		// TODO: This is a pretty ugly hack to workaround races during closing
		// in-memory etcd server in unit tests - see #18928 for more details.
		// We should get rid of it as soon as we have a proper fix - etcd clients
		// have overwritten transport counting opened connections (probably by
		// overwriting Dial function) and termination function waiting for all
		// connections to be closed and stopping accepting new ones.
		time.Sleep(250 * time.Millisecond)
		for _, hs := range m.hss {
			hs.CloseClientConnections()
			hs.Close()
		}
		if err := os.RemoveAll(m.ServerConfig.DataDir); err != nil {
			t.Fatal(err)
		}
		if len(m.CertificatesDir) > 0 {
			if err := os.RemoveAll(m.CertificatesDir); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// NewEtcdTestClientServer DEPRECATED creates a new client and server for testing
func NewEtcdTestClientServer(t *testing.T) *EtcdTestServer {
	server := configureTestCluster(t, "foo", true)
	err := server.launch(t)
	if err != nil {
		t.Fatalf("Failed to start etcd server error=%v", err)
		return nil
	}

	cfg := etcd.Config{
		Endpoints: server.ClientURLs.StringSlice(),
		Transport: newHttpTransport(t, server.CertFile, server.KeyFile, server.CAFile),
	}
	server.Client, err = etcd.New(cfg)
	if err != nil {
		server.Terminate(t)
		t.Fatalf("Unexpected error in NewEtcdTestClientServer (%v)", err)
		return nil
	}
	if err := server.waitUntilUp(); err != nil {
		server.Terminate(t)
		t.Fatalf("Unexpected error in waitUntilUp (%v)", err)
		return nil
	}
	return server
}

// NewUnsecuredEtcdTestClientServer DEPRECATED creates a new client and server for testing
func NewUnsecuredEtcdTestClientServer(t *testing.T) *EtcdTestServer {
	server := configureTestCluster(t, "foo", false)
	err := server.launch(t)
	if err != nil {
		t.Fatalf("Failed to start etcd server error=%v", err)
		return nil
	}
	cfg := etcd.Config{
		Endpoints: server.ClientURLs.StringSlice(),
		Transport: newHttpTransport(t, server.CertFile, server.KeyFile, server.CAFile),
	}
	server.Client, err = etcd.New(cfg)
	if err != nil {
		t.Errorf("Unexpected error in NewUnsecuredEtcdTestClientServer (%v)", err)
		server.Terminate(t)
		return nil
	}
	if err := server.waitUntilUp(); err != nil {
		t.Errorf("Unexpected error in waitUntilUp (%v)", err)
		server.Terminate(t)
		return nil
	}
	return server
}

// NewEtcd3TestClientServer creates a new client and server for testing
func NewUnsecuredEtcd3TestClientServer(t *testing.T, scheme *runtime.Scheme) (*EtcdTestServer, *storagebackend.Config) {
	server := &EtcdTestServer{
		v3Cluster: integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1}),
	}
	server.V3Client = server.v3Cluster.RandClient()
	config := &storagebackend.Config{
		Type:                     "etcd3",
		Prefix:                   etcdtest.PathPrefix(),
		ServerList:               server.V3Client.Endpoints(),
		DeserializationCacheSize: etcdtest.DeserializationCacheSize,
		Copier: scheme,
		Paging: true,
	}
	return server, config
}
