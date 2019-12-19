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

package etcd3

import (
	"context"
	"testing"

	"go.etcd.io/etcd/clientv3"
	etcdrpc "go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes"
	"go.etcd.io/etcd/integration"
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

	putResp1, err := client.Put(ctx, "/somekey", "data2")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	_, _, err = compact(ctx, client, 0, putResp1.Header.Revision)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	obj, err := client.Get(ctx, "/somekey", clientv3.WithRev(putResp.Header.Revision))
	if err != etcdrpc.ErrCompacted {
		t.Errorf("Expecting ErrCompacted, but get=%v err=%v", obj, err)
	}
}

// TestCompactConflict tests that two compactors (Let's use C1, C2) are trying to compact etcd cluster with the same
// logical time.
// - C1 compacts first. It will succeed.
// - C2 compacts after. It will fail. But it will get latest logical time, which should be larger by one.
func TestCompactConflict(t *testing.T) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	client := cluster.RandClient()
	ctx := context.Background()

	putResp, err := client.Put(ctx, "/somekey", "data")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Compact first. It would do the compaction and return compact time which is incremented by 1.
	curTime, _, err := compact(ctx, client, 0, putResp.Header.Revision)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	if curTime != 1 {
		t.Errorf("Expect current logical time = 1, get = %v", curTime)
	}

	// Compact again with the same parameters. It won't do compaction but return the latest compact time.
	curTime2, _, err := compact(ctx, client, 0, putResp.Header.Revision)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	if curTime != curTime2 {
		t.Errorf("Unexpected curTime (%v) != curTime2 (%v)", curTime, curTime2)
	}
}
