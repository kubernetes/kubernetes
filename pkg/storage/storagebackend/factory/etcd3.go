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
	"strings"

	"github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"

	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd3"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

func newETCD3Storage(c storagebackend.Config) (storage.Interface, error) {
	endpoints := c.ServerList
	for i, s := range endpoints {
		endpoints[i] = strings.TrimLeft(s, "http://")
	}
	cfg := clientv3.Config{
		Endpoints: endpoints,
	}
	client, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}
	etcd3.StartCompactor(context.Background(), client)
	return etcd3.New(client, c.Codec, c.Prefix), nil
}
