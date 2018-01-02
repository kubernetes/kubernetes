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

package integration

import (
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
)

// TestV3MaintenanceHashInflight ensures inflight Hash call
// to embedded being-stopped EtcdServer does not trigger panic.
func TestV3MaintenanceHashInflight(t *testing.T) {
	defer testutil.AfterTest(t)
	clus := NewClusterV3(t, &ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.RandClient()
	mvc := toGRPC(cli).Maintenance
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)

	donec := make(chan struct{})
	go func() {
		defer close(donec)
		mvc.Hash(ctx, &pb.HashRequest{}, grpc.FailFast(false))
	}()

	clus.Members[0].s.HardStop()
	cancel()

	<-donec
}
