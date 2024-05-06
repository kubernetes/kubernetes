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
	"testing"

	clientv3 "go.etcd.io/etcd/client/v3"

	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// EtcdTestServer encapsulates the datastructures needed to start local instance for testing
type EtcdTestServer struct {
	V3Client *clientv3.Client
}

func (e *EtcdTestServer) Terminate(t *testing.T) {
	// no-op, server termination moved to test cleanup
}

// NewUnsecuredEtcd3TestClientServer creates a new client and server for testing
func NewUnsecuredEtcd3TestClientServer(t *testing.T) (*EtcdTestServer, *storagebackend.Config) {
	server := &EtcdTestServer{}
	server.V3Client = testserver.RunEtcd(t, nil)
	config := &storagebackend.Config{
		Type:   "etcd3",
		Prefix: PathPrefix(),
		Transport: storagebackend.TransportConfig{
			ServerList: server.V3Client.Endpoints(),
		},
	}
	return server, config
}
