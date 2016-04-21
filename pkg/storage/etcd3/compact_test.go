/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package etcd3

import (
	"testing"

	"github.com/coreos/etcd/clientv3"
	etcdrpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/integration"
	"golang.org/x/net/context"
)

func TestCompact(t *testing.T) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	client := cluster.RandClient()
	ctx := context.Background()

	putResp, err := client.Put(ctx, "/somekey", "data")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	_, err = compact(ctx, client, putResp.Header.Revision)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	_, err = client.Get(ctx, "/somekey", clientv3.WithRev(putResp.Header.Revision))
	if err != etcdrpc.ErrCompacted {
		t.Errorf("Expecting ErrCompacted, but get=%v", err)
	}
}
