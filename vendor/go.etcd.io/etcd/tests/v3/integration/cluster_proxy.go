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

//go:build cluster_proxy
// +build cluster_proxy

package integration

import (
	"context"
	"sync"

	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/namespace"
	"go.etcd.io/etcd/server/v3/proxy/grpcproxy"
	"go.etcd.io/etcd/server/v3/proxy/grpcproxy/adapter"
	"go.uber.org/zap"
)

const ThroughProxy = true

var (
	pmu     sync.Mutex
	proxies map[*clientv3.Client]grpcClientProxy = make(map[*clientv3.Client]grpcClientProxy)
)

const proxyNamespace = "proxy-namespace"

type grpcClientProxy struct {
	ctx       context.Context
	ctxCancel func()
	grpc      grpcAPI
	wdonec    <-chan struct{}
	kvdonec   <-chan struct{}
	lpdonec   <-chan struct{}
}

func toGRPC(c *clientv3.Client) grpcAPI {
	pmu.Lock()
	defer pmu.Unlock()

	// dedicated context bound to 'grpc-proxy' lifetype
	// (so in practice lifetime of the client connection to the proxy).
	// TODO: Refactor to a separate clientv3.Client instance instead of the context alone.
	ctx, ctxCancel := context.WithCancel(context.WithValue(context.TODO(), "_name", "grpcProxyContext"))

	lg := c.GetLogger()

	if v, ok := proxies[c]; ok {
		return v.grpc
	}

	// test namespacing proxy
	c.KV = namespace.NewKV(c.KV, proxyNamespace)
	c.Watcher = namespace.NewWatcher(c.Watcher, proxyNamespace)
	c.Lease = namespace.NewLease(c.Lease, proxyNamespace)
	// test coalescing/caching proxy
	kvp, kvpch := grpcproxy.NewKvProxy(c)
	wp, wpch := grpcproxy.NewWatchProxy(ctx, lg, c)
	lp, lpch := grpcproxy.NewLeaseProxy(ctx, c)
	mp := grpcproxy.NewMaintenanceProxy(c)
	clp, _ := grpcproxy.NewClusterProxy(lg, c, "", "") // without registering proxy URLs
	authp := grpcproxy.NewAuthProxy(c)
	lockp := grpcproxy.NewLockProxy(c)
	electp := grpcproxy.NewElectionProxy(c)

	grpc := grpcAPI{
		adapter.ClusterServerToClusterClient(clp),
		adapter.KvServerToKvClient(kvp),
		adapter.LeaseServerToLeaseClient(lp),
		adapter.WatchServerToWatchClient(wp),
		adapter.MaintenanceServerToMaintenanceClient(mp),
		adapter.AuthServerToAuthClient(authp),
		adapter.LockServerToLockClient(lockp),
		adapter.ElectionServerToElectionClient(electp),
	}
	proxies[c] = grpcClientProxy{ctx: ctx, ctxCancel: ctxCancel, grpc: grpc, wdonec: wpch, kvdonec: kvpch, lpdonec: lpch}
	return grpc
}

type proxyCloser struct {
	clientv3.Watcher
	proxyCtxCancel func()
	wdonec         <-chan struct{}
	kvdonec        <-chan struct{}
	lclose         func()
	lpdonec        <-chan struct{}
}

func (pc *proxyCloser) Close() error {
	pc.proxyCtxCancel()
	<-pc.kvdonec
	err := pc.Watcher.Close()
	<-pc.wdonec
	pc.lclose()
	<-pc.lpdonec
	return err
}

func newClientV3(cfg clientv3.Config, lg *zap.Logger) (*clientv3.Client, error) {
	cfg.Logger = lg
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
		Watcher:        clientv3.NewWatchFromWatchClient(rpc.Watch, c),
		wdonec:         proxies[c].wdonec,
		kvdonec:        proxies[c].kvdonec,
		lclose:         func() { lc.Close() },
		lpdonec:        proxies[c].lpdonec,
		proxyCtxCancel: proxies[c].ctxCancel,
	}
	pmu.Unlock()
	return c, nil
}
