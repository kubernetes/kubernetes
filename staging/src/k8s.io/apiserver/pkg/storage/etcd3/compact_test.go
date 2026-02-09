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
	c := NewCompactor(client, time.Minute, clock, nil)
	t.Cleanup(c.Stop)
	waitForClockWaiters(t, clock)

	t.Log("First compaction cycle saves revision before first write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)
	compactRev := c.CompactRevision()
	if compactRev != 0 {
		t.Errorf("CompactRevision()=%d, expected %d", compactRev, 0)
	}

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
	compactRev = c.CompactRevision()
	if compactRev != putResp.Header.Revision-1 {
		t.Errorf("CompactRevision()=%d, expected %d", compactRev, 0)
	}

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
	compactRev = c.CompactRevision()
	if compactRev != putResp.Header.Revision+1 {
		t.Errorf("CompactRevision()=%d, expected %d", compactRev, putResp.Header.Revision)
	}

	assertNotCompacted(t, ctx, client, putResp1.Header.Revision)

	t.Log("Fourth compaction cycle compacts second write")
	clock.Step(time.Minute)
	waitForClockWaiters(t, clock)
	assertCompacted(t, ctx, client, putResp.Header.Revision)
	assertCompacted(t, ctx, client, putResp1.Header.Revision)
	compactRev = c.CompactRevision()
	if compactRev != putResp1.Header.Revision+1 {
		t.Errorf("CompactRevision()=%d, expected %d", compactRev, putResp1.Header.Revision)
	}
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

func TestCompactIntervalZero(t *testing.T) {
	client := testserver.RunEtcd(t, nil).Client
	clock := testingclock.NewFakeClock(time.Now())
	c := NewCompactor(client, 0, clock, nil)
	t.Cleanup(c.Stop)

	t.Log("Compact loop is disabled, no goroutine is waiting on clock")
	clockNoWaiters(t, clock)
	clock.Step(time.Minute)
	clockNoWaiters(t, clock)
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

func clockNoWaiters(t *testing.T, clock *testingclock.FakeClock) {
	t.Helper()
	for start := time.Now(); time.Since(start) < waitTimeout; {
		if clock.Waiters() != 0 {
			t.Fatal("waiter")
		}
		time.Sleep(waitDelay)
	}
	if clock.Waiters() != 0 {
		t.Fatal("waiter")
	}
}

// TestCompactConflict tests that multiple compactors are trying to compact etcd cluster with the same
// logical time.
// - C1 compacts on time 0. It will succeed.
// - C2 compacts on time 0. It will fail as this time was compacted. But it will get latest logical time, which should be larger by one.
// - C3 compacts on time 1. It will succeed.
func TestCompactConflict(t *testing.T) {
	client := testserver.RunEtcd(t, nil).Client
	ctx := context.Background()

	putResp, err := client.Put(ctx, "/somekey", "data")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	t.Log("First compact on time 0")
	wantCompactRev := putResp.Header.Revision
	curTime, curRev, compactRev, err := Compact(ctx, client, 0, wantCompactRev)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Compaction should succeed")
	if compactRev != wantCompactRev {
		t.Errorf("Expect compact revision = %d, get = %d", wantCompactRev, compactRev)
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
	curTime2, curRev2, compactRev2, err := Compact(ctx, client, 0, wantCompactRev+1)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Compaction should fail")
	if compactRev2 != wantCompactRev {
		t.Errorf("Expect compact revision = %d, get = %d", wantCompactRev, compactRev2)
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
	curTime3, curRev3, compactRev3, err := Compact(ctx, client, 1, wantCompactRev+1)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	t.Log("Compaction should succeed")
	wantCompactRev += 1
	if compactRev3 != wantCompactRev {
		t.Errorf("Expect compact revision = %d, get = %d", wantCompactRev, compactRev3)
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
