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
	"io/ioutil"
	"testing"

	"k8s.io/apiserver/pkg/storage/storagebackend"

	grpclogsettable "github.com/grpc-ecosystem/go-grpc-middleware/logging/settable"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/tests/v3/integration"
	"google.golang.org/grpc/grpclog"
)

var grpc_logger grpclogsettable.SettableLoggerV2

func init() {
	// override logger set up by etcd integration test package
	grpc_logger = grpclogsettable.ReplaceGrpcLoggerV2()
}

// EtcdTestServer encapsulates the datastructures needed to start local instance for testing
type EtcdTestServer struct {
	CertificatesDir string
	CertFile        string
	KeyFile         string
	CAFile          string

	// The following are lumped etcd3 test server params
	v3Cluster *integration.ClusterV3
	V3Client  *clientv3.Client
}

// Terminate will shutdown the running etcd server
func (m *EtcdTestServer) Terminate(t *testing.T) {
	m.v3Cluster.Terminate(t)
}

// NewUnsecuredEtcd3TestClientServer creates a new client and server for testing
func NewUnsecuredEtcd3TestClientServer(t *testing.T) (*EtcdTestServer, *storagebackend.Config) {
	integration.BeforeTestExternal(t)
	grpc_logger.Set(grpclog.NewLoggerV2(ioutil.Discard, ioutil.Discard, &testErrorWriter{t}))
	server := &EtcdTestServer{
		v3Cluster: integration.NewClusterV3(&noLogT{t}, &integration.ClusterConfig{Size: 1}),
	}
	server.V3Client = server.v3Cluster.RandClient()
	config := &storagebackend.Config{
		Type:   "etcd3",
		Prefix: PathPrefix(),
		Transport: storagebackend.TransportConfig{
			ServerList: server.V3Client.Endpoints(),
		},
		Paging: true,
	}
	return server, config
}

type noLogT struct {
	testing.TB
}

func (q *noLogT) Log(s ...interface{}) {
}
func (q *noLogT) Logf(s string, params ...interface{}) {
}

type testErrorWriter struct {
	testing.TB
}

func (t *testErrorWriter) Write(b []byte) (int, error) {
	t.TB.Error(string(b))
	return len(b), nil
}
