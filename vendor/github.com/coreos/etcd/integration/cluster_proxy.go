// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build cluster_proxy

package integration

import (
	"sync"

	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/proxy/grpcproxy"
)

var (
	pmu     sync.Mutex
	proxies map[*clientv3.Client]grpcAPI = make(map[*clientv3.Client]grpcAPI)
)

func toGRPC(c *clientv3.Client) grpcAPI {
	pmu.Lock()
	defer pmu.Unlock()

	if v, ok := proxies[c]; ok {
		return v
	}
	api := grpcAPI{
		pb.NewClusterClient(c.ActiveConnection()),
		grpcproxy.KvServerToKvClient(grpcproxy.NewKvProxy(c)),
		pb.NewLeaseClient(c.ActiveConnection()),
		grpcproxy.WatchServerToWatchClient(grpcproxy.NewWatchProxy(c)),
		pb.NewMaintenanceClient(c.ActiveConnection()),
	}
	proxies[c] = api
	return api
}

func newClientV3(cfg clientv3.Config) (*clientv3.Client, error) {
	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}
	rpc := toGRPC(c)
	c.KV = clientv3.NewKVFromKVClient(rpc.KV)
	c.Watcher = clientv3.NewWatchFromWatchClient(rpc.Watch)
	return c, nil
}
