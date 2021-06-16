// Copyright 2020 The etcd Authors
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

package integration

import (
	"log"
	"net/http"
	"sync"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/testutil"
	"go.etcd.io/etcd/client/pkg/v3/transport"
)

// Infrastructure to provision a single shared cluster for tests - only
// when its needed.
//
// See ./tests/integration/clientv3/examples/main_test.go for canonical usage.
// Please notice that the shared (LazyCluster's) state is preserved between
// testcases, so left-over state might has cross-testcase effects.
// Prefer dedicated clusters for substancial test-cases.

type LazyCluster interface {
	// EndpointsV2 - exposes connection points for client v2.
	// Calls to this method might initialize the cluster.
	EndpointsV2() []string

	// EndpointsV3 - exposes connection points for client v3.
	// Calls to this method might initialize the cluster.
	EndpointsV3() []string

	// Cluster - calls to this method might initialize the cluster.
	Cluster() *ClusterV3

	// Transport - call to this method might initialize the cluster.
	Transport() *http.Transport

	Terminate()

	TB() testutil.TB
}

type lazyCluster struct {
	cfg       ClusterConfig
	cluster   *ClusterV3
	transport *http.Transport
	once      sync.Once
	tb        testutil.TB
	closer    func()
}

// NewLazyCluster returns a new test cluster handler that gets created on the
// first call to GetEndpoints() or GetTransport()
func NewLazyCluster() LazyCluster {
	return NewLazyClusterWithConfig(ClusterConfig{Size: 1})
}

// NewLazyClusterWithConfig returns a new test cluster handler that gets created
// on the first call to GetEndpoints() or GetTransport()
func NewLazyClusterWithConfig(cfg ClusterConfig) LazyCluster {
	tb, closer := testutil.NewTestingTBProthesis("lazy_cluster")
	return &lazyCluster{cfg: cfg, tb: tb, closer: closer}
}

func (lc *lazyCluster) mustLazyInit() {
	lc.once.Do(func() {
		var err error
		lc.transport, err = transport.NewTransport(transport.TLSInfo{}, time.Second)
		if err != nil {
			log.Fatal(err)
		}
		lc.cluster = NewClusterV3(lc.tb, &lc.cfg)
	})
}

func (lc *lazyCluster) Terminate() {
	lc.tb.Logf("Terminating...")
	if lc != nil && lc.cluster != nil {
		lc.cluster.Terminate(nil)
		lc.cluster = nil
	}
	if lc.closer != nil {
		lc.tb.Logf("Closer...")
		lc.closer()
	}
}

func (lc *lazyCluster) EndpointsV2() []string {
	return []string{lc.Cluster().Members[0].URL()}
}

func (lc *lazyCluster) EndpointsV3() []string {
	return lc.Cluster().Client(0).Endpoints()
}

func (lc *lazyCluster) Cluster() *ClusterV3 {
	lc.mustLazyInit()
	return lc.cluster
}

func (lc *lazyCluster) Transport() *http.Transport {
	lc.mustLazyInit()
	return lc.transport
}

func (lc *lazyCluster) TB() testutil.TB {
	return lc.tb
}
