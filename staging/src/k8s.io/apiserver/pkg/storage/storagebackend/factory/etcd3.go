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

package factory

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	grpcprom "github.com/grpc-ecosystem/go-grpc-prometheus"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
)

// The short keepalive timeout and interval have been chosen to aggressively
// detect a failed etcd server without introducing much overhead.
const keepaliveTime = 30 * time.Second
const keepaliveTimeout = 10 * time.Second

// dialTimeout is the timeout for failing to establish a connection.
// It is set to 20 seconds as times shorter than that will cause TLS connections to fail
// on heavily loaded arm64 CPUs (issue #64649)
const dialTimeout = 20 * time.Second

func newETCD3HealthCheck(c storagebackend.Config) (func() error, error) {
	// constructing the etcd v3 client blocks and times out if etcd is not available.
	// retry in a loop in the background until we successfully create the client, storing the client or error encountered

	clientValue := &atomic.Value{}

	clientErrMsg := &atomic.Value{}
	clientErrMsg.Store("etcd client connection not yet established")

	go wait.PollUntil(time.Second, func() (bool, error) {
		client, err := newETCD3Client(c)
		if err != nil {
			clientErrMsg.Store(err.Error())
			return false, nil
		}
		clientValue.Store(client)
		clientErrMsg.Store("")
		return true, nil
	}, wait.NeverStop)

	return func() error {
		if errMsg := clientErrMsg.Load().(string); len(errMsg) > 0 {
			return fmt.Errorf(errMsg)
		}
		client := clientValue.Load().(*clientv3.Client)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if _, err := client.Cluster.MemberList(ctx); err != nil {
			return fmt.Errorf("error listing etcd members: %v", err)
		}
		return nil
	}, nil
}

func newETCD3Client(c storagebackend.Config) (*clientv3.Client, error) {
	tlsInfo := transport.TLSInfo{
		CertFile: c.CertFile,
		KeyFile:  c.KeyFile,
		CAFile:   c.CAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, err
	}
	// NOTE: Client relies on nil tlsConfig
	// for non-secure connections, update the implicit variable
	if len(c.CertFile) == 0 && len(c.KeyFile) == 0 && len(c.CAFile) == 0 {
		tlsConfig = nil
	}
	cfg := clientv3.Config{
		DialTimeout:          dialTimeout,
		DialKeepAliveTime:    keepaliveTime,
		DialKeepAliveTimeout: keepaliveTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithUnaryInterceptor(grpcprom.UnaryClientInterceptor),
			grpc.WithStreamInterceptor(grpcprom.StreamClientInterceptor),
		},
		Endpoints: c.ServerList,
		TLS:       tlsConfig,
	}
	client, err := clientv3.New(cfg)
	return client, err
}

func newETCD3Storage(c storagebackend.Config) (storage.Interface, DestroyFunc, error) {
	client, err := newETCD3Client(c)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	etcd3.StartCompactor(ctx, client, c.CompactionInterval)
	destroyFunc := func() {
		cancel()
		client.Close()
	}
	transformer := c.Transformer
	if transformer == nil {
		transformer = value.IdentityTransformer
	}
	if c.Quorum {
		return etcd3.New(client, c.Codec, c.Prefix, transformer, c.Paging), destroyFunc, nil
	}
	return etcd3.NewWithNoQuorumRead(client, c.Codec, c.Prefix, transformer, c.Paging), destroyFunc, nil
}
