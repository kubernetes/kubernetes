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
	"github.com/coreos/etcd/clientv3/namespace"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/proxy/grpcproxy"
	"github.com/coreos/etcd/proxy/grpcproxy/adapter"
)

var (
	pmu     sync.Mutex
	proxies map[*clientv3.Client]grpcClientProxy = make(map[*clientv3.Client]grpcClientProxy)
)

const proxyNamespace = "proxy-namespace"

type grpcClientProxy struct {
	grpc    grpcAPI
	wdonec  <-chan struct{}
	kvdonec <-chan struct{}
	lpdonec <-chan struct{}
}

func toGRPC(c *clientv3.Client) grpcAPI {
	pmu.Lock()
	defer pmu.Unlock()

	if v, ok := proxies[c]; ok {
		return v.grpc
	}

	// test namespacing proxy
	c.KV = namespace.NewKV(c.KV, proxyNamespace)
	c.Watcher = namespace.NewWatcher(c.Watcher, proxyNamespace)
	c.Lease = namespace.NewLease(c.Lease, proxyNamespace)
	// test coalescing/caching proxy
	kvp, kvpch := grpcproxy.NewKvProxy(c)
	wp, wpch := grpcproxy.NewWatchProxy(c)
	lp, lpch := grpcproxy.NewLeaseProxy(c)
	mp := grpcproxy.NewMaintenanceProxy(c)
	clp, _ := grpcproxy.NewClusterProxy(c, "", "") // without registering proxy URLs
	lockp := grpcproxy.NewLockProxy(c)
	electp := grpcproxy.NewElectionProxy(c)

	grpc := grpcAPI{
		adapter.ClusterServerToClusterClient(clp),
		adapter.KvServerToKvClient(kvp),
		adapter.LeaseServerToLeaseClient(lp),
		adapter.WatchServerToWatchClient(wp),
		adapter.MaintenanceServerToMaintenanceClient(mp),
		pb.NewAuthClient(c.ActiveConnection()),
		adapter.LockServerToLockClient(lockp),
		adapter.ElectionServerToElectionClient(electp),
	}
	proxies[c] = grpcClientProxy{grpc: grpc, wdonec: wpch, kvdonec: kvpch, lpdonec: lpch}
	return grpc
}

type proxyCloser struct {
	clientv3.Watcher
	wdonec  <-chan struct{}
	kvdonec <-chan struct{}
	lclose  func()
	lpdonec <-chan struct{}
}

func (pc *proxyCloser) Close() error {
	// client ctx is canceled before calling close, so kv and lp will close out
	<-pc.kvdonec
	err := pc.Watcher.Close()
	<-pc.wdonec
	pc.lclose()
	<-pc.lpdonec
	return err
}

func newClientV3(cfg clientv3.Config) (*clientv3.Client, error) {
	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}
	rpc := toGRPC(c)
	c.KV = clientv3.NewKVFromKVClient(rpc.KV, c)
	pmu.Lock()
	lc := c.Lease
	c.Lease = clientv3.NewLeaseFromLeaseClient(rpc.Lease, c, cfg.DialTimeout)
	c.Watcher = &proxyCloser{
		Watcher: clientv3.NewWatchFromWatchClient(rpc.Watch, c),
		wdonec:  proxies[c].wdonec,
		kvdonec: proxies[c].kvdonec,
		lclose:  func() { lc.Close() },
		lpdonec: proxies[c].lpdonec,
	}
	pmu.Unlock()
	return c, nil
}
