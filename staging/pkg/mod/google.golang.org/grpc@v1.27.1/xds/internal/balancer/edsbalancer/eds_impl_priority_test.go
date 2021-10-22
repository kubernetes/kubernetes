/*
 *
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
	"reflect"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	xdsclient "google.golang.org/grpc/xds/internal/client"
)

// When a high priority is ready, adding/removing lower locality doesn't cause
// changes.
//
// Init 0 and 1; 0 is up, use 0; add 2, use 0; remove 2, use 0.
func TestEDSPriority_HighPriorityReady(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with priorities [0, 1], each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh

	// p0 is ready.
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Test roundrobin with only p0 subconns.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		// t.Fatalf("want %v, got %v", want, err)
		t.Fatalf("want %v, got %v", want, err)
	}

	// Add p2, it shouldn't cause any udpates.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab2.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	clab2.AddLocality(testSubZones[2], 1, 2, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	select {
	case <-cc.newPickerCh:
		t.Fatalf("got unexpected new picker")
	case <-cc.newSubConnCh:
		t.Fatalf("got unexpected new SubConn")
	case <-cc.removeSubConnCh:
		t.Fatalf("got unexpected remove SubConn")
	case <-time.After(time.Millisecond * 100):
	}

	// Remove p2, no updates.
	clab3 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab3.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab3.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab3.Build()))

	select {
	case <-cc.newPickerCh:
		t.Fatalf("got unexpected new picker")
	case <-cc.newSubConnCh:
		t.Fatalf("got unexpected new SubConn")
	case <-cc.removeSubConnCh:
		t.Fatalf("got unexpected remove SubConn")
	case <-time.After(time.Millisecond * 100):
	}
}

// Lower priority is used when higher priority is not ready.
//
// Init 0 and 1; 0 is up, use 0; 0 is down, 1 is up, use 1; add 2, use 1; 1 is
// down, use 2; remove 2, use 1.
func TestEDSPriority_SwitchPriority(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with priorities [0, 1], each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh

	// p0 is ready.
	edsb.HandleSubConnStateChange(sc0, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc0, connectivity.Ready)

	// Test roundrobin with only p0 subconns.
	p0 := <-cc.newPickerCh
	want := []balancer.SubConn{sc0}
	if err := isRoundRobin(want, subConnFromPicker(p0)); err != nil {
		// t.Fatalf("want %v, got %v", want, err)
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn down 0, 1 is used.
	edsb.HandleSubConnStateChange(sc0, connectivity.TransientFailure)
	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[1]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Test pick with 1.
	p1 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc1) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}

	// Add p2, it shouldn't cause any udpates.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab2.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	clab2.AddLocality(testSubZones[2], 1, 2, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	select {
	case <-cc.newPickerCh:
		t.Fatalf("got unexpected new picker")
	case <-cc.newSubConnCh:
		t.Fatalf("got unexpected new SubConn")
	case <-cc.removeSubConnCh:
		t.Fatalf("got unexpected remove SubConn")
	case <-time.After(time.Millisecond * 100):
	}

	// Turn down 1, use 2
	edsb.HandleSubConnStateChange(sc1, connectivity.TransientFailure)
	addrs2 := <-cc.newSubConnAddrsCh
	if got, want := addrs2[0].Addr, testEndpointAddrs[2]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test pick with 2.
	p2 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p2.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc2) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc2)
		}
	}

	// Remove 2, use 1.
	clab3 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab3.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab3.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab3.Build()))

	// p2 SubConns are removed.
	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc2) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc2, scToRemove)
	}

	// Should get an update with 1's old picker, to override 2's old picker.
	p3 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := p3.Pick(balancer.PickInfo{}); err != balancer.ErrTransientFailure {
			t.Fatalf("want pick error %v, got %v", balancer.ErrTransientFailure, err)
		}
	}
}

// Add a lower priority while the higher priority is down.
//
// Init 0 and 1; 0 and 1 both down; add 2, use 2.
func TestEDSPriority_HigherDownWhileAddingLower(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with different priorities, each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh

	// Turn down 0, 1 is used.
	edsb.HandleSubConnStateChange(sc0, connectivity.TransientFailure)
	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[1]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh
	// Turn down 1, pick should error.
	edsb.HandleSubConnStateChange(sc1, connectivity.TransientFailure)

	// Test pick failure.
	pFail := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := pFail.Pick(balancer.PickInfo{}); err != balancer.ErrTransientFailure {
			t.Fatalf("want pick error %v, got %v", balancer.ErrTransientFailure, err)
		}
	}

	// Add p2, it should create a new SubConn.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab2.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	clab2.AddLocality(testSubZones[2], 1, 2, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	addrs2 := <-cc.newSubConnAddrsCh
	if got, want := addrs2[0].Addr, testEndpointAddrs[2]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test pick with 2.
	p2 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p2.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc2) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc2)
		}
	}
}

// When a higher priority becomes available, all lower priorities are closed.
//
// Init 0,1,2; 0 and 1 down, use 2; 0 up, close 1 and 2.
func TestEDSPriority_HigherReadyCloseAllLower(t *testing.T) {
	defer time.Sleep(10 * time.Millisecond)

	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with priorities [0,1,2], each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	clab1.AddLocality(testSubZones[2], 1, 2, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh

	// Turn down 0, 1 is used.
	edsb.HandleSubConnStateChange(sc0, connectivity.TransientFailure)
	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[1]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh
	// Turn down 1, 2 is used.
	edsb.HandleSubConnStateChange(sc1, connectivity.TransientFailure)
	addrs2 := <-cc.newSubConnAddrsCh
	if got, want := addrs2[0].Addr, testEndpointAddrs[2]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test pick with 2.
	p2 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p2.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc2) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc2)
		}
	}

	// When 0 becomes ready, 0 should be used, 1 and 2 should all be closed.
	edsb.HandleSubConnStateChange(sc0, connectivity.Ready)

	// sc1 and sc2 should be removed.
	//
	// With localities caching, the lower priorities are closed after a timeout,
	// in goroutines. The order is no longer guaranteed.
	scToRemove := []balancer.SubConn{<-cc.removeSubConnCh, <-cc.removeSubConnCh}
	if !(reflect.DeepEqual(scToRemove[0], sc1) && reflect.DeepEqual(scToRemove[1], sc2)) &&
		!(reflect.DeepEqual(scToRemove[0], sc2) && reflect.DeepEqual(scToRemove[1], sc1)) {
		t.Errorf("RemoveSubConn, want [%v, %v], got %v", sc1, sc2, scToRemove)
	}

	// Test pick with 0.
	p0 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p0.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc0) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc0)
		}
	}
}

// At init, start the next lower priority after timeout if the higher priority
// doesn't get ready.
//
// Init 0,1; 0 is not ready (in connecting), after timeout, use 1.
func TestEDSPriority_InitTimeout(t *testing.T) {
	const testPriorityInitTimeout = time.Second
	defer func() func() {
		old := defaultPriorityInitTimeout
		defaultPriorityInitTimeout = testPriorityInitTimeout
		return func() {
			defaultPriorityInitTimeout = old
		}
	}()()

	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with different priorities, each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh

	// Keep 0 in connecting, 1 will be used after init timeout.
	edsb.HandleSubConnStateChange(sc0, connectivity.Connecting)

	// Make sure new SubConn is created before timeout.
	select {
	case <-time.After(testPriorityInitTimeout * 3 / 4):
	case <-cc.newSubConnAddrsCh:
		t.Fatalf("Got a new SubConn too early (Within timeout). Expect a new SubConn only after timeout")
	}

	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[1]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh

	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Test pick with 1.
	p1 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc1) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}
}

// Add localities to existing priorities.
//
//  - start with 2 locality with p0 and p1
//  - add localities to existing p0 and p1
func TestEDSPriority_MultipleLocalities(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with different priorities, each with one backend.
	clab0 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab0.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab0.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab0.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc0, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc0, connectivity.Ready)

	// Test roundrobin with only p0 subconns.
	p0 := <-cc.newPickerCh
	want := []balancer.SubConn{sc0}
	if err := isRoundRobin(want, subConnFromPicker(p0)); err != nil {
		// t.Fatalf("want %v, got %v", want, err)
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn down p0 subconns, p1 subconns will be created.
	edsb.HandleSubConnStateChange(sc0, connectivity.TransientFailure)

	addrs1 := <-cc.newSubConnAddrsCh
	if got, want := addrs1[0].Addr, testEndpointAddrs[1]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Test roundrobin with only p1 subconns.
	p1 := <-cc.newPickerCh
	want = []balancer.SubConn{sc1}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		// t.Fatalf("want %v, got %v", want, err)
		t.Fatalf("want %v, got %v", want, err)
	}

	// Reconnect p0 subconns, p1 subconn will be closed.
	edsb.HandleSubConnStateChange(sc0, connectivity.Ready)

	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc1) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc1, scToRemove)
	}

	// Test roundrobin with only p0 subconns.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{sc0}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Add two localities, with two priorities, with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	clab1.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:3], nil)
	clab1.AddLocality(testSubZones[3], 1, 1, testEndpointAddrs[3:4], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	addrs2 := <-cc.newSubConnAddrsCh
	if got, want := addrs2[0].Addr, testEndpointAddrs[2]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin with only two p0 subconns.
	p3 := <-cc.newPickerCh
	want = []balancer.SubConn{sc0, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Turn down p0 subconns, p1 subconns will be created.
	edsb.HandleSubConnStateChange(sc0, connectivity.TransientFailure)
	edsb.HandleSubConnStateChange(sc2, connectivity.TransientFailure)

	sc3 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc3, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc3, connectivity.Ready)
	sc4 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc4, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc4, connectivity.Ready)

	// Test roundrobin with only p1 subconns.
	p4 := <-cc.newPickerCh
	want = []balancer.SubConn{sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p4)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

// EDS removes all localities, and re-adds them.
func TestEDSPriority_RemovesAllLocalities(t *testing.T) {
	const testPriorityInitTimeout = time.Second
	defer func() func() {
		old := defaultPriorityInitTimeout
		defaultPriorityInitTimeout = testPriorityInitTimeout
		return func() {
			defaultPriorityInitTimeout = old
		}
	}()()

	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, with different priorities, each with one backend.
	clab0 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab0.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab0.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab0.Build()))

	addrs0 := <-cc.newSubConnAddrsCh
	if got, want := addrs0[0].Addr, testEndpointAddrs[0]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc0 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc0, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc0, connectivity.Ready)

	// Test roundrobin with only p0 subconns.
	p0 := <-cc.newPickerCh
	want := []balancer.SubConn{sc0}
	if err := isRoundRobin(want, subConnFromPicker(p0)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove all priorities.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	// p0 subconn should be removed.
	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc0) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc0, scToRemove)
	}

	// Test pick return TransientFailure.
	pFail := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if _, err := pFail.Pick(balancer.PickInfo{}); err != balancer.ErrTransientFailure {
			t.Fatalf("want pick error %v, got %v", balancer.ErrTransientFailure, err)
		}
	}

	// Re-add two localities, with previous priorities, but different backends.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[2:3], nil)
	clab2.AddLocality(testSubZones[1], 1, 1, testEndpointAddrs[3:4], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	addrs01 := <-cc.newSubConnAddrsCh
	if got, want := addrs01[0].Addr, testEndpointAddrs[2]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc01 := <-cc.newSubConnCh

	// Don't send any update to p0, so to not override the old state of p0.
	// Later, connect to p1 and then remove p1. This will fallback to p0, and
	// will send p0's old picker if they are not correctly removed.

	// p1 will be used after priority init timeout.
	addrs11 := <-cc.newSubConnAddrsCh
	if got, want := addrs11[0].Addr, testEndpointAddrs[3]; got != want {
		t.Fatalf("sc is created with addr %v, want %v", got, want)
	}
	sc11 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc11, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc11, connectivity.Ready)

	// Test roundrobin with only p1 subconns.
	p1 := <-cc.newPickerCh
	want = []balancer.SubConn{sc11}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove p1 from EDS, to fallback to p0.
	clab3 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab3.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab3.Build()))

	// p1 subconn should be removed.
	scToRemove1 := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove1, sc11) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc11, scToRemove1)
	}

	// Test pick return TransientFailure.
	pFail1 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		if scst, err := pFail1.Pick(balancer.PickInfo{}); err != balancer.ErrNoSubConnAvailable {
			t.Fatalf("want pick error _, %v, got %v, _ ,%v", balancer.ErrTransientFailure, scst, err)
		}
	}

	// Send an ready update for the p0 sc that was received when re-adding
	// localities to EDS.
	edsb.HandleSubConnStateChange(sc01, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc01, connectivity.Ready)

	// Test roundrobin with only p0 subconns.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{sc01}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	select {
	case <-cc.newPickerCh:
		t.Fatalf("got unexpected new picker")
	case <-cc.newSubConnCh:
		t.Fatalf("got unexpected new SubConn")
	case <-cc.removeSubConnCh:
		t.Fatalf("got unexpected remove SubConn")
	case <-time.After(time.Millisecond * 100):
	}
}

func TestPriorityType(t *testing.T) {
	p0 := newPriorityType(0)
	p1 := newPriorityType(1)
	p2 := newPriorityType(2)

	if !p0.higherThan(p1) || !p0.higherThan(p2) {
		t.Errorf("want p0 to be higher than p1 and p2, got p0>p1: %v, p0>p2: %v", !p0.higherThan(p1), !p0.higherThan(p2))
	}
	if !p1.lowerThan(p0) || !p1.higherThan(p2) {
		t.Errorf("want p1 to be between p0 and p2, got p1<p0: %v, p1>p2: %v", !p1.lowerThan(p0), !p1.higherThan(p2))
	}
	if !p2.lowerThan(p0) || !p2.lowerThan(p1) {
		t.Errorf("want p2 to be lower than p0 and p1, got p2<p0: %v, p2<p1: %v", !p2.lowerThan(p0), !p2.lowerThan(p1))
	}

	if got := p1.equal(p0.nextLower()); !got {
		t.Errorf("want p1 to be equal to p0's next lower, got p1==p0.nextLower: %v", got)
	}

	if got := p1.equal(newPriorityType(1)); !got {
		t.Errorf("want p1 to be equal to priority with value 1, got p1==1: %v", got)
	}
}
