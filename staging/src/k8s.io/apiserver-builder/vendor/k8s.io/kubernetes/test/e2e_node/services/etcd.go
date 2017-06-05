/*
Copyright 2016 The Kubernetes Authors.

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

package services

import (
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/pkg/capnslog"
	"github.com/golang/glog"
)

// TODO(random-liu): Add service interface to manage services with the same behaviour.

func init() {
	// github.com/coreos/etcd/etcdserver/api package is too spammy, set the log level to NOTICE.
	capnslog.MustRepoLogger("github.com/coreos/etcd/etcdserver/api").SetRepoLogLevel(capnslog.NOTICE)
}

// All following configurations are got from etcd source code.
// TODO(random-liu): Use embed.NewConfig after etcd3 is supported.
const (
	etcdName           = "etcd"
	clientURLStr       = "http://localhost:4001" // clientURL has listener created and handles etcd API traffic
	peerURLStr         = "http://localhost:7001" // peerURL does't have listener created, it is used to pass Etcd validation
	snapCount          = etcdserver.DefaultSnapCount
	maxSnapFiles       = 5
	maxWALFiles        = 5
	tickMs             = 100
	electionTicks      = 10
	etcdHealthCheckURL = clientURLStr + "/v2/keys/" // Trailing slash is required,
)

// EtcdServer is a server which manages etcd.
type EtcdServer struct {
	*etcdserver.EtcdServer
	config       *etcdserver.ServerConfig
	clientListen net.Listener
}

// NewEtcd creates a new default etcd server using 'dataDir' for persistence.
func NewEtcd(dataDir string) *EtcdServer {
	clientURLs, err := types.NewURLs([]string{clientURLStr})
	if err != nil {
		glog.Fatalf("Failed to parse client url %q: %v", clientURLStr, err)
	}
	peerURLs, err := types.NewURLs([]string{peerURLStr})
	if err != nil {
		glog.Fatalf("Failed to parse peer url %q: %v", peerURLStr, err)
	}

	config := &etcdserver.ServerConfig{
		Name:               etcdName,
		ClientURLs:         clientURLs,
		PeerURLs:           peerURLs,
		DataDir:            dataDir,
		InitialPeerURLsMap: map[string]types.URLs{etcdName: peerURLs},
		NewCluster:         true,
		SnapCount:          snapCount,
		MaxSnapFiles:       maxSnapFiles,
		MaxWALFiles:        maxWALFiles,
		TickMs:             tickMs,
		ElectionTicks:      electionTicks,
	}

	return &EtcdServer{
		config: config,
	}
}

// Start starts the etcd server and listening for client connections
func (e *EtcdServer) Start() error {
	var err error
	e.EtcdServer, err = etcdserver.NewServer(e.config)
	if err != nil {
		return err
	}
	// create client listener, there should be only one url
	e.clientListen, err = createListener(e.config.ClientURLs[0])
	if err != nil {
		return err
	}

	// start etcd
	e.EtcdServer.Start()

	// setup client listener
	ch := v2http.NewClientHandler(e.EtcdServer, e.config.ReqTimeout())
	errCh := make(chan error)
	go func(l net.Listener) {
		defer close(errCh)
		srv := &http.Server{
			Handler:     ch,
			ReadTimeout: 5 * time.Minute,
		}
		// Serve always returns a non-nil error.
		errCh <- srv.Serve(l)
	}(e.clientListen)

	err = readinessCheck("etcd", []string{etcdHealthCheckURL}, errCh)
	if err != nil {
		return err
	}
	return nil
}

// Stop closes all connections and stops the Etcd server
func (e *EtcdServer) Stop() error {
	if e.EtcdServer != nil {
		e.EtcdServer.Stop()
	}
	if e.clientListen != nil {
		err := e.clientListen.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

// Name returns the server's unique name
func (e *EtcdServer) Name() string {
	return etcdName
}

func createListener(url url.URL) (net.Listener, error) {
	l, err := net.Listen("tcp", url.Host)
	if err != nil {
		return nil, err
	}
	l, err = transport.NewKeepAliveListener(l, url.Scheme, &tls.Config{})
	if err != nil {
		return nil, err
	}
	return l, nil
}

func getEtcdClientURL() string {
	return clientURLStr
}

func getEtcdHealthCheckURL() string {
	return etcdHealthCheckURL
}
