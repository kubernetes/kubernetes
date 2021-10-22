/*
 * Copyright 2019 gRPC authors.
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
 */

package edsbalancer

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/xds/internal"
	orcapb "google.golang.org/grpc/xds/internal/proto/udpa/data/orca/v1"
)

var (
	rrBuilder        = balancer.Get(roundrobin.Name)
	testBalancerIDs  = []internal.Locality{{Region: "b1"}, {Region: "b2"}, {Region: "b3"}}
	testBackendAddrs []resolver.Address
)

const testBackendAddrsCount = 12

func init() {
	for i := 0; i < testBackendAddrsCount; i++ {
		testBackendAddrs = append(testBackendAddrs, resolver.Address{Addr: fmt.Sprintf("%d.%d.%d.%d:%d", i, i, i, i, i)})
	}

	// Disable caching for all tests. It will be re-enabled in caching specific
	// tests.
	defaultSubBalancerCloseTimeout = time.Millisecond
}

func subConnFromPicker(p balancer.V2Picker) func() balancer.SubConn {
	return func() balancer.SubConn {
		scst, _ := p.Pick(balancer.PickInfo{})
		return scst.SubConn
	}
}

// 1 balancer, 1 backend -> 2 backends -> 1 backend.
func TestBalancerGroup_OneRR_AddRemoveBackend(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add one balancer to group.
	bg.add(testBalancerIDs[0], 1, rrBuilder)
	// Send one resolved address.
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:1])

	// Send subconn state change.
	sc1 := <-cc.newSubConnCh
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)

	// Test pick with one backend.
	p1 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc1) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}

	// Send two addresses.
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	// Expect one new subconn, send state update.
	sc2 := <-cc.newSubConnCh
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin pick.
	p2 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove the first address.
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[1:2])
	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc1) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc1, scToRemove)
	}
	bg.handleSubConnStateChange(scToRemove, connectivity.Shutdown)

	// Test pick with only the second subconn.
	p3 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSC, _ := p3.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSC.SubConn, sc2) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSC, sc2)
		}
	}
}

// 2 balancers, each with 1 backend.
func TestBalancerGroup_TwoRR_OneBackend(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add two balancers to group and send one resolved address to both
	// balancers.
	bg.add(testBalancerIDs[0], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:1])
	sc1 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[0:1])
	sc2 := <-cc.newSubConnCh

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

// 2 balancers, each with more than 1 backends.
func TestBalancerGroup_TwoRR_MoreBackends(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add two balancers to group and send one resolved address to both
	// balancers.
	bg.add(testBalancerIDs[0], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	sc1 := <-cc.newSubConnCh
	sc2 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])
	sc3 := <-cc.newSubConnCh
	sc4 := <-cc.newSubConnCh

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)
	bg.handleSubConnStateChange(sc3, connectivity.Connecting)
	bg.handleSubConnStateChange(sc3, connectivity.Ready)
	bg.handleSubConnStateChange(sc4, connectivity.Connecting)
	bg.handleSubConnStateChange(sc4, connectivity.Ready)

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn sc2's connection down, should be RR between balancers.
	bg.handleSubConnStateChange(sc2, connectivity.TransientFailure)
	p2 := <-cc.newPickerCh
	// Expect two sc1's in the result, because balancer1 will be picked twice,
	// but there's only one sc in it.
	want = []balancer.SubConn{sc1, sc1, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove sc3's addresses.
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[3:4])
	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc3) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc3, scToRemove)
	}
	bg.handleSubConnStateChange(scToRemove, connectivity.Shutdown)
	p3 := <-cc.newPickerCh
	want = []balancer.SubConn{sc1, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn sc1's connection down.
	bg.handleSubConnStateChange(sc1, connectivity.TransientFailure)
	p4 := <-cc.newPickerCh
	want = []balancer.SubConn{sc4}
	if err := isRoundRobin(want, subConnFromPicker(p4)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn last connection to connecting.
	bg.handleSubConnStateChange(sc4, connectivity.Connecting)
	p5 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := p5.Pick(balancer.PickInfo{}); err != balancer.ErrNoSubConnAvailable {
			t.Fatalf("want pick error %v, got %v", balancer.ErrNoSubConnAvailable, err)
		}
	}

	// Turn all connections down.
	bg.handleSubConnStateChange(sc4, connectivity.TransientFailure)
	p6 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := p6.Pick(balancer.PickInfo{}); err != balancer.ErrTransientFailure {
			t.Fatalf("want pick error %v, got %v", balancer.ErrTransientFailure, err)
		}
	}
}

// 2 balancers with different weights.
func TestBalancerGroup_TwoRR_DifferentWeight_MoreBackends(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add two balancers to group and send two resolved addresses to both
	// balancers.
	bg.add(testBalancerIDs[0], 2, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	sc1 := <-cc.newSubConnCh
	sc2 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])
	sc3 := <-cc.newSubConnCh
	sc4 := <-cc.newSubConnCh

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)
	bg.handleSubConnStateChange(sc3, connectivity.Connecting)
	bg.handleSubConnStateChange(sc3, connectivity.Ready)
	bg.handleSubConnStateChange(sc4, connectivity.Connecting)
	bg.handleSubConnStateChange(sc4, connectivity.Ready)

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc1, sc2, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

// totally 3 balancers, add/remove balancer.
func TestBalancerGroup_ThreeRR_RemoveBalancer(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add three balancers to group and send one resolved address to both
	// balancers.
	bg.add(testBalancerIDs[0], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:1])
	sc1 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[1:2])
	sc2 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[2], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[2], testBackendAddrs[1:2])
	sc3 := <-cc.newSubConnCh

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)
	bg.handleSubConnStateChange(sc3, connectivity.Connecting)
	bg.handleSubConnStateChange(sc3, connectivity.Ready)

	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2, sc3}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove the second balancer, while the others two are ready.
	bg.remove(testBalancerIDs[1])
	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc2) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc2, scToRemove)
	}
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{sc1, sc3}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// move balancer 3 into transient failure.
	bg.handleSubConnStateChange(sc3, connectivity.TransientFailure)
	// Remove the first balancer, while the third is transient failure.
	bg.remove(testBalancerIDs[0])
	scToRemove = <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc1) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc1, scToRemove)
	}
	p3 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := p3.Pick(balancer.PickInfo{}); err != balancer.ErrTransientFailure {
			t.Fatalf("want pick error %v, got %v", balancer.ErrTransientFailure, err)
		}
	}
}

// 2 balancers, change balancer weight.
func TestBalancerGroup_TwoRR_ChangeWeight_MoreBackends(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)
	bg.start()

	// Add two balancers to group and send two resolved addresses to both
	// balancers.
	bg.add(testBalancerIDs[0], 2, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	sc1 := <-cc.newSubConnCh
	sc2 := <-cc.newSubConnCh

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])
	sc3 := <-cc.newSubConnCh
	sc4 := <-cc.newSubConnCh

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)
	bg.handleSubConnStateChange(sc3, connectivity.Connecting)
	bg.handleSubConnStateChange(sc3, connectivity.Ready)
	bg.handleSubConnStateChange(sc4, connectivity.Connecting)
	bg.handleSubConnStateChange(sc4, connectivity.Ready)

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc1, sc2, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	bg.changeWeight(testBalancerIDs[0], 3)

	// Test roundrobin with new weight.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{sc1, sc1, sc1, sc2, sc2, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

func TestBalancerGroup_LoadReport(t *testing.T) {
	testLoadStore := newTestLoadStore()

	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, testLoadStore)
	bg.start()

	backendToBalancerID := make(map[balancer.SubConn]internal.Locality)

	// Add two balancers to group and send two resolved addresses to both
	// balancers.
	bg.add(testBalancerIDs[0], 2, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	sc1 := <-cc.newSubConnCh
	sc2 := <-cc.newSubConnCh
	backendToBalancerID[sc1] = testBalancerIDs[0]
	backendToBalancerID[sc2] = testBalancerIDs[0]

	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])
	sc3 := <-cc.newSubConnCh
	sc4 := <-cc.newSubConnCh
	backendToBalancerID[sc3] = testBalancerIDs[1]
	backendToBalancerID[sc4] = testBalancerIDs[1]

	// Send state changes for both subconns.
	bg.handleSubConnStateChange(sc1, connectivity.Connecting)
	bg.handleSubConnStateChange(sc1, connectivity.Ready)
	bg.handleSubConnStateChange(sc2, connectivity.Connecting)
	bg.handleSubConnStateChange(sc2, connectivity.Ready)
	bg.handleSubConnStateChange(sc3, connectivity.Connecting)
	bg.handleSubConnStateChange(sc3, connectivity.Ready)
	bg.handleSubConnStateChange(sc4, connectivity.Connecting)
	bg.handleSubConnStateChange(sc4, connectivity.Ready)

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	var (
		wantStart []internal.Locality
		wantEnd   []internal.Locality
		wantCost  []testServerLoad
	)
	for i := 0; i < 10; i++ {
		scst, _ := p1.Pick(balancer.PickInfo{})
		locality := backendToBalancerID[scst.SubConn]
		wantStart = append(wantStart, locality)
		if scst.Done != nil && scst.SubConn != sc1 {
			scst.Done(balancer.DoneInfo{
				ServerLoad: &orcapb.OrcaLoadReport{
					CpuUtilization: 10,
					MemUtilization: 5,
					RequestCost:    map[string]float64{"pic": 3.14},
					Utilization:    map[string]float64{"piu": 3.14},
				},
			})
			wantEnd = append(wantEnd, locality)
			wantCost = append(wantCost,
				testServerLoad{name: serverLoadCPUName, d: 10},
				testServerLoad{name: serverLoadMemoryName, d: 5},
				testServerLoad{name: "pic", d: 3.14},
				testServerLoad{name: "piu", d: 3.14})
		}
	}

	if !reflect.DeepEqual(testLoadStore.callsStarted, wantStart) {
		t.Fatalf("want started: %v, got: %v", testLoadStore.callsStarted, wantStart)
	}
	if !reflect.DeepEqual(testLoadStore.callsEnded, wantEnd) {
		t.Fatalf("want ended: %v, got: %v", testLoadStore.callsEnded, wantEnd)
	}
	if !reflect.DeepEqual(testLoadStore.callsCost, wantCost) {
		t.Fatalf("want cost: %v, got: %v", testLoadStore.callsCost, wantCost)
	}
}

// Create a new balancer group, add balancer and backends, but not start.
// - b1, weight 2, backends [0,1]
// - b2, weight 1, backends [2,3]
// Start the balancer group and check behavior.
//
// Close the balancer group, call add/remove/change weight/change address.
// - b2, weight 3, backends [0,3]
// - b3, weight 1, backends [1,2]
// Start the balancer group again and check for behavior.
func TestBalancerGroup_start_close(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)

	// Add two balancers to group and send two resolved addresses to both
	// balancers.
	bg.add(testBalancerIDs[0], 2, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])

	bg.start()

	m1 := make(map[resolver.Address]balancer.SubConn)
	for i := 0; i < 4; i++ {
		addrs := <-cc.newSubConnAddrsCh
		sc := <-cc.newSubConnCh
		m1[addrs[0]] = sc
		bg.handleSubConnStateChange(sc, connectivity.Connecting)
		bg.handleSubConnStateChange(sc, connectivity.Ready)
	}

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{
		m1[testBackendAddrs[0]], m1[testBackendAddrs[0]],
		m1[testBackendAddrs[1]], m1[testBackendAddrs[1]],
		m1[testBackendAddrs[2]], m1[testBackendAddrs[3]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	bg.close()
	for i := 0; i < 4; i++ {
		bg.handleSubConnStateChange(<-cc.removeSubConnCh, connectivity.Shutdown)
	}

	// Add b3, weight 1, backends [1,2].
	bg.add(testBalancerIDs[2], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[2], testBackendAddrs[1:3])

	// Remove b1.
	bg.remove(testBalancerIDs[0])

	// Update b2 to weight 3, backends [0,3].
	bg.changeWeight(testBalancerIDs[1], 3)
	bg.handleResolvedAddrs(testBalancerIDs[1], append([]resolver.Address(nil), testBackendAddrs[0], testBackendAddrs[3]))

	bg.start()

	m2 := make(map[resolver.Address]balancer.SubConn)
	for i := 0; i < 4; i++ {
		addrs := <-cc.newSubConnAddrsCh
		sc := <-cc.newSubConnCh
		m2[addrs[0]] = sc
		bg.handleSubConnStateChange(sc, connectivity.Connecting)
		bg.handleSubConnStateChange(sc, connectivity.Ready)
	}

	// Test roundrobin on the last picker.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{
		m2[testBackendAddrs[0]], m2[testBackendAddrs[0]], m2[testBackendAddrs[0]],
		m2[testBackendAddrs[3]], m2[testBackendAddrs[3]], m2[testBackendAddrs[3]],
		m2[testBackendAddrs[1]], m2[testBackendAddrs[2]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

// Test that balancer group start() doesn't deadlock if the balancer calls back
// into balancer group inline when it gets an update.
//
// The potential deadlock can happen if we
//  - hold a lock and send updates to balancer (e.g. update resolved addresses)
//  - the balancer calls back (NewSubConn or update picker) in line
// The callback will try to hold hte same lock again, which will cause a
// deadlock.
//
// This test starts the balancer group with a test balancer, will updates picker
// whenever it gets an address update. It's expected that start() doesn't block
// because of deadlock.
func TestBalancerGroup_start_close_deadlock(t *testing.T) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)

	bg.add(testBalancerIDs[0], 2, &testConstBalancerBuilder{})
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	bg.add(testBalancerIDs[1], 1, &testConstBalancerBuilder{})
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])

	bg.start()
}

func replaceDefaultSubBalancerCloseTimeout(n time.Duration) func() {
	old := defaultSubBalancerCloseTimeout
	defaultSubBalancerCloseTimeout = n
	return func() { defaultSubBalancerCloseTimeout = old }
}

// initBalancerGroupForCachingTest creates a balancer group, and initialize it
// to be ready for caching tests.
//
// Two rr balancers are added to bg, each with 2 ready subConns. A sub-balancer
// is removed later, so the balancer group returned has one sub-balancer in its
// own map, and one sub-balancer in cache.
func initBalancerGroupForCachingTest(t *testing.T) (*balancerGroup, *testClientConn, map[resolver.Address]balancer.SubConn) {
	cc := newTestClientConn(t)
	bg := newBalancerGroup(cc, nil)

	// Add two balancers to group and send two resolved addresses to both
	// balancers.
	bg.add(testBalancerIDs[0], 2, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[0], testBackendAddrs[0:2])
	bg.add(testBalancerIDs[1], 1, rrBuilder)
	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[2:4])

	bg.start()

	m1 := make(map[resolver.Address]balancer.SubConn)
	for i := 0; i < 4; i++ {
		addrs := <-cc.newSubConnAddrsCh
		sc := <-cc.newSubConnCh
		m1[addrs[0]] = sc
		bg.handleSubConnStateChange(sc, connectivity.Connecting)
		bg.handleSubConnStateChange(sc, connectivity.Ready)
	}

	// Test roundrobin on the last picker.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{
		m1[testBackendAddrs[0]], m1[testBackendAddrs[0]],
		m1[testBackendAddrs[1]], m1[testBackendAddrs[1]],
		m1[testBackendAddrs[2]], m1[testBackendAddrs[3]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	bg.remove(testBalancerIDs[1])
	// Don't wait for SubConns to be removed after close, because they are only
	// removed after close timeout.
	for i := 0; i < 10; i++ {
		select {
		case <-cc.removeSubConnCh:
			t.Fatalf("Got request to remove subconn, want no remove subconn (because subconns were still in cache)")
		default:
		}
		time.Sleep(time.Millisecond)
	}
	// Test roundrobin on the with only sub-balancer0.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{
		m1[testBackendAddrs[0]], m1[testBackendAddrs[1]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	return bg, cc, m1
}

// Test that if a sub-balancer is removed, and re-added within close timeout,
// the subConns won't be re-created.
func TestBalancerGroup_locality_caching(t *testing.T) {
	defer replaceDefaultSubBalancerCloseTimeout(10 * time.Second)()
	bg, cc, addrToSC := initBalancerGroupForCachingTest(t)

	// Turn down subconn for addr2, shouldn't get picker update because
	// sub-balancer1 was removed.
	bg.handleSubConnStateChange(addrToSC[testBackendAddrs[2]], connectivity.TransientFailure)
	for i := 0; i < 10; i++ {
		select {
		case <-cc.newPickerCh:
			t.Fatalf("Got new picker, want no new picker (because the sub-balancer was removed)")
		default:
		}
		time.Sleep(time.Millisecond)
	}

	// Sleep, but sleep less then close timeout.
	time.Sleep(time.Millisecond * 100)

	// Re-add sub-balancer-1, because subconns were in cache, no new subconns
	// should be created. But a new picker will still be generated, with subconn
	// states update to date.
	bg.add(testBalancerIDs[1], 1, rrBuilder)

	p3 := <-cc.newPickerCh
	want := []balancer.SubConn{
		addrToSC[testBackendAddrs[0]], addrToSC[testBackendAddrs[0]],
		addrToSC[testBackendAddrs[1]], addrToSC[testBackendAddrs[1]],
		// addr2 is down, b2 only has addr3 in READY state.
		addrToSC[testBackendAddrs[3]], addrToSC[testBackendAddrs[3]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	for i := 0; i < 10; i++ {
		select {
		case <-cc.newSubConnAddrsCh:
			t.Fatalf("Got new subconn, want no new subconn (because subconns were still in cache)")
		default:
		}
		time.Sleep(time.Millisecond * 10)
	}
}

// Sub-balancers are put in cache when they are removed. If balancer group is
// closed within close timeout, all subconns should still be rmeoved
// immediately.
func TestBalancerGroup_locality_caching_close_group(t *testing.T) {
	defer replaceDefaultSubBalancerCloseTimeout(10 * time.Second)()
	bg, cc, addrToSC := initBalancerGroupForCachingTest(t)

	bg.close()
	// The balancer group is closed. The subconns should be removed immediately.
	removeTimeout := time.After(time.Millisecond * 500)
	scToRemove := map[balancer.SubConn]int{
		addrToSC[testBackendAddrs[0]]: 1,
		addrToSC[testBackendAddrs[1]]: 1,
		addrToSC[testBackendAddrs[2]]: 1,
		addrToSC[testBackendAddrs[3]]: 1,
	}
	for i := 0; i < len(scToRemove); i++ {
		select {
		case sc := <-cc.removeSubConnCh:
			c := scToRemove[sc]
			if c == 0 {
				t.Fatalf("Got removeSubConn for %v when there's %d remove expected", sc, c)
			}
			scToRemove[sc] = c - 1
		case <-removeTimeout:
			t.Fatalf("timeout waiting for subConns (from balancer in cache) to be removed")
		}
	}
}

// Sub-balancers in cache will be closed if not re-added within timeout, and
// subConns will be removed.
func TestBalancerGroup_locality_caching_not_readd_within_timeout(t *testing.T) {
	defer replaceDefaultSubBalancerCloseTimeout(time.Second)()
	_, cc, addrToSC := initBalancerGroupForCachingTest(t)

	// The sub-balancer is not re-added withtin timeout. The subconns should be
	// removed.
	removeTimeout := time.After(defaultSubBalancerCloseTimeout)
	scToRemove := map[balancer.SubConn]int{
		addrToSC[testBackendAddrs[2]]: 1,
		addrToSC[testBackendAddrs[3]]: 1,
	}
	for i := 0; i < len(scToRemove); i++ {
		select {
		case sc := <-cc.removeSubConnCh:
			c := scToRemove[sc]
			if c == 0 {
				t.Fatalf("Got removeSubConn for %v when there's %d remove expected", sc, c)
			}
			scToRemove[sc] = c - 1
		case <-removeTimeout:
			t.Fatalf("timeout waiting for subConns (from balancer in cache) to be removed")
		}
	}
}

// Wrap the rr builder, so it behaves the same, but has a different pointer.
type noopBalancerBuilderWrapper struct {
	balancer.Builder
}

// After removing a sub-balancer, re-add with same ID, but different balancer
// builder. Old subconns should be removed, and new subconns should be created.
func TestBalancerGroup_locality_caching_readd_with_different_builder(t *testing.T) {
	defer replaceDefaultSubBalancerCloseTimeout(10 * time.Second)()
	bg, cc, addrToSC := initBalancerGroupForCachingTest(t)

	// Re-add sub-balancer-1, but with a different balancer builder. The
	// sub-balancer was still in cache, but cann't be reused. This should cause
	// old sub-balancer's subconns to be removed immediately, and new subconns
	// to be created.
	bg.add(testBalancerIDs[1], 1, &noopBalancerBuilderWrapper{rrBuilder})

	// The cached sub-balancer should be closed, and the subconns should be
	// removed immediately.
	removeTimeout := time.After(time.Millisecond * 500)
	scToRemove := map[balancer.SubConn]int{
		addrToSC[testBackendAddrs[2]]: 1,
		addrToSC[testBackendAddrs[3]]: 1,
	}
	for i := 0; i < len(scToRemove); i++ {
		select {
		case sc := <-cc.removeSubConnCh:
			c := scToRemove[sc]
			if c == 0 {
				t.Fatalf("Got removeSubConn for %v when there's %d remove expected", sc, c)
			}
			scToRemove[sc] = c - 1
		case <-removeTimeout:
			t.Fatalf("timeout waiting for subConns (from balancer in cache) to be removed")
		}
	}

	bg.handleResolvedAddrs(testBalancerIDs[1], testBackendAddrs[4:6])

	newSCTimeout := time.After(time.Millisecond * 500)
	scToAdd := map[resolver.Address]int{
		testBackendAddrs[4]: 1,
		testBackendAddrs[5]: 1,
	}
	for i := 0; i < len(scToAdd); i++ {
		select {
		case addr := <-cc.newSubConnAddrsCh:
			c := scToAdd[addr[0]]
			if c == 0 {
				t.Fatalf("Got newSubConn for %v when there's %d new expected", addr, c)
			}
			scToAdd[addr[0]] = c - 1
			sc := <-cc.newSubConnCh
			addrToSC[addr[0]] = sc
			bg.handleSubConnStateChange(sc, connectivity.Connecting)
			bg.handleSubConnStateChange(sc, connectivity.Ready)
		case <-newSCTimeout:
			t.Fatalf("timeout waiting for subConns (from new sub-balancer) to be newed")
		}
	}

	// Test roundrobin on the new picker.
	p3 := <-cc.newPickerCh
	want := []balancer.SubConn{
		addrToSC[testBackendAddrs[0]], addrToSC[testBackendAddrs[0]],
		addrToSC[testBackendAddrs[1]], addrToSC[testBackendAddrs[1]],
		addrToSC[testBackendAddrs[4]], addrToSC[testBackendAddrs[5]],
	}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}
