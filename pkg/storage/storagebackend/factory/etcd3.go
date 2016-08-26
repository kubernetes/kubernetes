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
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd3"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"golang.org/x/net/context"
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

	cfg := clientv3.Config{
		Endpoints: c.ServerList,
		TLS:       tlsConfig,
	}
	client, err := clientv3.New(cfg)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	etcd3.StartCompactor(ctx, client)
	destroyFunc := func() {
		cancel()
		client.Close()
	}
	return etcd3.New(client, c.Codec, c.Prefix), destroyFunc, nil
}
