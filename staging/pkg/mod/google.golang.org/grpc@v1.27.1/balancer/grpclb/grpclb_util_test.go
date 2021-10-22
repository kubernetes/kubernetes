/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpclb

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/resolver"
)

type mockSubConn struct {
	balancer.SubConn
}

type mockClientConn struct {
	balancer.ClientConn

	mu       sync.Mutex
	subConns map[balancer.SubConn]resolver.Address
}

func newMockClientConn() *mockClientConn {
	return &mockClientConn{
		subConns: make(map[balancer.SubConn]resolver.Address),
	}
}

func (mcc *mockClientConn) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	sc := &mockSubConn{}
	mcc.mu.Lock()
	defer mcc.mu.Unlock()
	mcc.subConns[sc] = addrs[0]
	return sc, nil
}

func (mcc *mockClientConn) RemoveSubConn(sc balancer.SubConn) {
	mcc.mu.Lock()
	defer mcc.mu.Unlock()
	delete(mcc.subConns, sc)
}

const testCacheTimeout = 100 * time.Millisecond

func checkMockCC(mcc *mockClientConn, scLen int) error {
	mcc.mu.Lock()
	defer mcc.mu.Unlock()
	if len(mcc.subConns) != scLen {
		return fmt.Errorf("mcc = %+v, want len(mcc.subConns) = %v", mcc.subConns, scLen)
	}
	return nil
}

func checkCacheCC(ccc *lbCacheClientConn, sccLen, sctaLen int) error {
	ccc.mu.Lock()
	defer ccc.mu.Unlock()
	if len(ccc.subConnCache) != sccLen {
		return fmt.Errorf("ccc = %+v, want len(ccc.subConnCache) = %v", ccc.subConnCache, sccLen)
	}
	if len(ccc.subConnToAddr) != sctaLen {
		return fmt.Errorf("ccc = %+v, want len(ccc.subConnToAddr) = %v", ccc.subConnToAddr, sctaLen)
	}
	return nil
}

// Test that SubConn won't be immediately removed.
func TestLBCacheClientConnExpire(t *testing.T) {
	mcc := newMockClientConn()
	if err := checkMockCC(mcc, 0); err != nil {
		t.Fatal(err)
	}

	ccc := newLBCacheClientConn(mcc)
	ccc.timeout = testCacheTimeout
	if err := checkCacheCC(ccc, 0, 0); err != nil {
		t.Fatal(err)
	}

	sc, _ := ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	ccc.RemoveSubConn(sc)
	// One subconn in MockCC before timeout.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// One subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Should all become empty after timeout.
	var err error
	for i := 0; i < 2; i++ {
		time.Sleep(testCacheTimeout)
		err = checkMockCC(mcc, 0)
		if err != nil {
			continue
		}
		err = checkCacheCC(ccc, 0, 0)
		if err != nil {
			continue
		}
	}
	if err != nil {
		t.Fatal(err)
	}
}

// Test that NewSubConn with the same address of a SubConn being removed will
// reuse the SubConn and cancel the removing.
func TestLBCacheClientConnReuse(t *testing.T) {
	mcc := newMockClientConn()
	if err := checkMockCC(mcc, 0); err != nil {
		t.Fatal(err)
	}

	ccc := newLBCacheClientConn(mcc)
	ccc.timeout = testCacheTimeout
	if err := checkCacheCC(ccc, 0, 0); err != nil {
		t.Fatal(err)
	}

	sc, _ := ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	ccc.RemoveSubConn(sc)
	// One subconn in MockCC before timeout.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// One subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Recreate the old subconn, this should cancel the deleting process.
	sc, _ = ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	var err error
	// Should not become empty after 2*timeout.
	time.Sleep(2 * testCacheTimeout)
	err = checkMockCC(mcc, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = checkCacheCC(ccc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Call remove again, will delete after timeout.
	ccc.RemoveSubConn(sc)
	// One subconn in MockCC before timeout.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// One subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Should all become empty after timeout.
	for i := 0; i < 2; i++ {
		time.Sleep(testCacheTimeout)
		err = checkMockCC(mcc, 0)
		if err != nil {
			continue
		}
		err = checkCacheCC(ccc, 0, 0)
		if err != nil {
			continue
		}
	}
	if err != nil {
		t.Fatal(err)
	}
}

// Test that if the timer to remove a SubConn fires at the same time NewSubConn
// cancels the timer, it doesn't cause deadlock.
func TestLBCache_RemoveTimer_New_Race(t *testing.T) {
	mcc := newMockClientConn()
	if err := checkMockCC(mcc, 0); err != nil {
		t.Fatal(err)
	}

	ccc := newLBCacheClientConn(mcc)
	ccc.timeout = time.Nanosecond
	if err := checkCacheCC(ccc, 0, 0); err != nil {
		t.Fatal(err)
	}

	sc, _ := ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})

	go func() {
		for i := 0; i < 1000; i++ {
			// Remove starts a timer with 1 ns timeout, the NewSubConn will race
			// with with the timer.
			ccc.RemoveSubConn(sc)
			sc, _ = ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
		}
		close(done)
	}()

	select {
	case <-time.After(time.Second):
		t.Fatalf("Test didn't finish within 1 second. Deadlock")
	case <-done:
	}
}
