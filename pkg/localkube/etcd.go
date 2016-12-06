/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package localkube

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
)

const (
	EtcdName = "etcd"
)

var (
	// EtcdClientURLs have listeners created and handle etcd API traffic
	KubeEtcdClientURLs = []string{"http://localhost:2379"}

	// EtcdPeerURLs don't have listeners created for them, they are used to pass Etcd validation
	KubeEtcdPeerURLs = []string{"http://localhost:2380"}
)

// Etcd is a Server which manages an Etcd cluster
type EtcdServer struct {
	*etcdserver.EtcdServer
	config          *etcdserver.ServerConfig
	clientListeners []net.Listener
}

// NewEtcd creates a new default etcd Server using 'dataDir' for persistence. Panics if could not be configured.
func (lk LocalkubeServer) NewEtcd(clientURLStrs, peerURLStrs []string, name, dataDirectory string) (*EtcdServer, error) {
	clientURLs, err := types.NewURLs(clientURLStrs)
	if err != nil {
		return nil, err
	}

	peerURLs, err := types.NewURLs(peerURLStrs)
	if err != nil {
		return nil, err
	}

	urlsMap := map[string]types.URLs{
		name: peerURLs,
	}

	config := &etcdserver.ServerConfig{
		Name:               name,
		ClientURLs:         clientURLs,
		PeerURLs:           peerURLs,
		DataDir:            dataDirectory,
		InitialPeerURLsMap: urlsMap,

		NewCluster: true,

		SnapCount:     etcdserver.DefaultSnapCount,
		MaxSnapFiles:  5,
		MaxWALFiles:   5,
		TickMs:        100,
		ElectionTicks: 10,
	}

	lk.SetExtraConfigForComponent(EtcdName, &config)

	return &EtcdServer{
		config: config,
	}, nil
}

// Starts starts the etcd server and listening for client connections
func (e *EtcdServer) Start() {
	var err error
	e.EtcdServer, err = etcdserver.NewServer(e.config)
	if err != nil {
		msg := fmt.Sprintf("Etcd config error: %v", err)
		panic(msg)
	}

	// create client listeners
	e.clientListeners = createListenersOrPanic(e.config.ClientURLs)

	// start etcd
	e.EtcdServer.Start()

	// setup client listeners
	ch := v2http.NewClientHandler(e.EtcdServer, e.requestTimeout())
	for _, l := range e.clientListeners {
		go func(l net.Listener) {
			srv := &http.Server{
				Handler:     ch,
				ReadTimeout: 5 * time.Minute,
			}
			panic(srv.Serve(l))
		}(l)
	}
}

// Stop closes all connections and stops the Etcd server
func (e *EtcdServer) Stop() {
	if e.EtcdServer != nil {
		e.EtcdServer.Stop()
	}

	for _, l := range e.clientListeners {
		l.Close()
	}
}

// Name returns the servers unique name
func (EtcdServer) Name() string {
	return EtcdName
}

func (e *EtcdServer) requestTimeout() time.Duration {
	// from github.com/coreos/etcd/etcdserver/config.go
	return 5*time.Second + 2*time.Duration(e.config.ElectionTicks)*time.Duration(e.config.TickMs)*time.Millisecond
}

func createListenersOrPanic(urls types.URLs) (listeners []net.Listener) {
	for _, url := range urls {
		l, err := net.Listen("tcp", url.Host)
		if err != nil {
			panic(err)
		}

		l, err = transport.NewKeepAliveListener(l, url.Scheme, &tls.Config{})
		if err != nil {
			panic(err)
		}

		listeners = append(listeners, l)
	}
	return listeners
}
