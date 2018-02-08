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
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"golang.org/x/net/context"

	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
)

// The short keepalive timeout and interval have been chosen to aggressively
// detect a failed etcd server without introducing much overhead.
var (
	keepaliveTime    = 30 * time.Second
	keepaliveTimeout = 10 * time.Second
)

func newETCD3Storage(c storagebackend.Config) (storage.Interface, DestroyFunc, error) {
	tlsInfo := transport.TLSInfo{
		CertFile: c.CertFile,
		KeyFile:  c.KeyFile,
		CAFile:   c.CAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, nil, err
	}
	// NOTE: Client relies on nil tlsConfig
	// for non-secure connections, update the implicit variable
	if len(c.CertFile) == 0 && len(c.KeyFile) == 0 && len(c.CAFile) == 0 {
		tlsConfig = nil
	}
	cfg := clientv3.Config{
		DialKeepAliveTime:    keepaliveTime,
		DialKeepAliveTimeout: keepaliveTimeout,
		Endpoints:            c.ServerList,
		TLS:                  tlsConfig,
	}
	client, err := clientv3.New(cfg)
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
