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
	"sync"
	"testing"
	"time"

	etcdrpc "go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	clientv3 "go.etcd.io/etcd/client/v3"

	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	waitDelay   = time.Millisecond
	waitTimeout = 100 * waitDelay
)

func TestCompact(t *testing.T) {
	client := testserver.RunEtcd(t, nil).Client
	ctx := context.Background()
	clock := testingclock.NewFakeClock(time.Now())
	wg := sync.WaitGroup{}
	defer wg.Wait()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	wg.Add(1)
	go func() {
		defer wg.Done()
		compactor(ctx, client, clock, time.Minute)
	}()
	waitForClockWaiters(t, clock)

	t.Log("First compaction cycle saves revision before first write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)

	t.Log("First write")
	putResp, err := client.Put(ctx, "/somekey", "data")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	assertNotCompacted(t, ctx, client, putResp.Header.Revision)

	t.Log("Second compaction cycle compacts before first write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)
	assertNotCompacted(t, ctx, client, putResp.Header.Revision)

	t.Log("Create second revision")
	putResp1, err := client.Put(ctx, "/somekey", "data2")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	assertNotCompacted(t, ctx, client, putResp1.Header.Revision)

	t.Log("Third compaction cycle compacts revision after first write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)
	assertCompacted(t, ctx, client, putResp.Header.Revision)

	assertNotCompacted(t, ctx, client, putResp1.Header.Revision)

	t.Log("Fourth compaction cycle compacts second write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)
	assertCompacted(t, ctx, client, putResp.Header.Revision)
	assertCompacted(t, ctx, client, putResp1.Header.Revision)
}

func assertCompacted(t *testing.T, ctx context.Context, client *clientv3.Client, rev int64) {
	t.Helper()
	_, err := client.Get(ctx, "/somekey", clientv3.WithRev(rev))
	if err != etcdrpc.ErrCompacted {
		t.Errorf("Expecting rev %d compacted, but err=%v", rev, err)
	}
}

func assertNotCompacted(t *testing.T, ctx context.Context, client *clientv3.Client, rev int64) {
	t.Helper()
	_, err := client.Get(ctx, "/somekey", clientv3.WithRev(rev))
	if err != nil {
		t.Errorf("Get on rev %d failed: %v", rev, err)
	}
}

func waitForClockWaiters(t *testing.T, clock *testingclock.FakeClock) {
	t.Helper()
	for start := time.Now(); time.Since(start) < waitTimeout; {
		if clock.HasWaiters() {
			return
		}
		time.Sleep(waitDelay)
	}
	t.Fatal("No waiters")
}

// TestCompactConflict tests that multiple compactors are trying to compact etcd cluster with the same
// logical time.
// - C1 compacts on time 0. It will succeed.
// - C2 compacts on time 0. It will fail as this time was compacted. But it will get latest logical time, which should be larger by one.
// - C3 compacts on time 1. It will succeed
func TestCompactConflict(t *testing.T) {
	client := testserver.RunEtcd(t, nil).Client
	ctx := context.Background()

	putResp, err := client.Put(ctx, "/somekey", "data")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	t.Log("First compact on time 0")
	wantCompactRev := putResp.Header.Revision
	curTime, curRev, err := compact(ctx, client, 0, wantCompactRev)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Current time should increase by 1")
	if curTime != 1 {
		t.Errorf("Expect current logical time = 1, get = %v", curTime)
	}
	t.Log("Current revision should increase by 1")
	wantCurrentRev := putResp.Header.Revision + 1
	if curRev != wantCurrentRev {
		t.Errorf("Expect current revision = %d, get = %d", wantCurrentRev, curRev)
	}

	t.Log("Second compact on time 0")
	curTime2, curRev2, err := compact(ctx, client, 0, wantCompactRev+1)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Should return same time as from the first compacty")
	if curTime != curTime2 {
		t.Errorf("Unexpected curTime (%v) != curTime2 (%v)", curTime, curTime2)
	}
	t.Log("Current revision should stay the same")
	if curRev2 != wantCurrentRev {
		t.Errorf("Expect current revision = %d, get = %d", wantCurrentRev, curRev2)
	}

	t.Log("Third compact on time 1")
	curTime3, curRev3, err := compact(ctx, client, 1, wantCompactRev+1)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Current time should increase by 1")
	if curTime3 != 2 {
		t.Errorf("Expect current logical time = 2, get = %v", curTime3)
	}
	t.Log("Current revision should increase by 1")
	wantCurrentRev += 1
	if curRev3 != wantCurrentRev {
		t.Errorf("Expect current revision = %d, get = %d", wantCurrentRev, curRev3)
	}
}
