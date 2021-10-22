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
	"sort"
	"testing"
	"time"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/xds/internal"
	xdsclient "google.golang.org/grpc/xds/internal/client"
)

var (
	testClusterNames  = []string{"test-cluster-1", "test-cluster-2"}
	testSubZones      = []string{"I", "II", "III", "IV"}
	testEndpointAddrs []string
)

func init() {
	for i := 0; i < testBackendAddrsCount; i++ {
		testEndpointAddrs = append(testEndpointAddrs, fmt.Sprintf("%d.%d.%d.%d:%d", i, i, i, i, i))
	}
}

// One locality
//  - add backend
//  - remove backend
//  - replace backend
//  - change drop rate
func TestEDS_OneLocality(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// One locality with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Pick with only the first backend.
	p1 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc1) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}

	// The same locality, add one more backend.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin with two subconns.
	p2 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// The same locality, delete first backend.
	clab3 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab3.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab3.Build()))

	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc1) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc1, scToRemove)
	}
	edsb.HandleSubConnStateChange(scToRemove, connectivity.Shutdown)

	// Test pick with only the second subconn.
	p3 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p3.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc2) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc2)
		}
	}

	// The same locality, replace backend.
	clab4 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab4.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab4.Build()))

	sc3 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc3, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc3, connectivity.Ready)
	scToRemove = <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc2) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc2, scToRemove)
	}
	edsb.HandleSubConnStateChange(scToRemove, connectivity.Shutdown)

	// Test pick with only the third subconn.
	p4 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p4.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(gotSCSt.SubConn, sc3) {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc3)
		}
	}

	// The same locality, different drop rate, dropping 50%.
	clab5 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], []uint32{50})
	clab5.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab5.Build()))

	// Picks with drops.
	p5 := <-cc.newPickerCh
	for i := 0; i < 100; i++ {
		_, err := p5.Pick(balancer.PickInfo{})
		// TODO: the dropping algorithm needs a design. When the dropping algorithm
		// is fixed, this test also needs fix.
		if i < 50 && err == nil {
			t.Errorf("The first 50%% picks should be drops, got error <nil>")
		} else if i > 50 && err != nil {
			t.Errorf("The second 50%% picks should be non-drops, got error %v", err)
		}
	}
}

// 2 locality
//  - start with 2 locality
//  - add locality
//  - remove locality
//  - address change for the <not-the-first> locality
//  - update locality weight
func TestEDS_TwoLocalities(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))
	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)

	// Add the second locality later to make sure sc2 belongs to the second
	// locality. Otherwise the test is flaky because of a map is used in EDS to
	// keep localities.
	clab1.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin with two subconns.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Add another locality, with one backend.
	clab2 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab2.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab2.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	clab2.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab2.Build()))

	sc3 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc3, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc3, connectivity.Ready)

	// Test roundrobin with three subconns.
	p2 := <-cc.newPickerCh
	want = []balancer.SubConn{sc1, sc2, sc3}
	if err := isRoundRobin(want, subConnFromPicker(p2)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Remove first locality.
	clab3 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab3.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	clab3.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:3], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab3.Build()))

	scToRemove := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove, sc1) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc1, scToRemove)
	}
	edsb.HandleSubConnStateChange(scToRemove, connectivity.Shutdown)

	// Test pick with two subconns (without the first one).
	p3 := <-cc.newPickerCh
	want = []balancer.SubConn{sc2, sc3}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Add a backend to the last locality.
	clab4 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab4.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	clab4.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:4], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab4.Build()))

	sc4 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc4, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc4, connectivity.Ready)

	// Test pick with two subconns (without the first one).
	p4 := <-cc.newPickerCh
	// Locality-1 will be picked twice, and locality-2 will be picked twice.
	// Locality-1 contains only sc2, locality-2 contains sc3 and sc4. So expect
	// two sc2's and sc3, sc4.
	want = []balancer.SubConn{sc2, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p4)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Change weight of the locality[1].
	clab5 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab5.AddLocality(testSubZones[1], 2, 0, testEndpointAddrs[1:2], nil)
	clab5.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:4], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab5.Build()))

	// Test pick with two subconns different locality weight.
	p5 := <-cc.newPickerCh
	// Locality-1 will be picked four times, and locality-2 will be picked twice
	// (weight 2 and 1). Locality-1 contains only sc2, locality-2 contains sc3 and
	// sc4. So expect four sc2's and sc3, sc4.
	want = []balancer.SubConn{sc2, sc2, sc2, sc2, sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p5)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	// Change weight of the locality[1] to 0, it should never be picked.
	clab6 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab6.AddLocality(testSubZones[1], 0, 0, testEndpointAddrs[1:2], nil)
	clab6.AddLocality(testSubZones[2], 1, 0, testEndpointAddrs[2:4], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab6.Build()))

	// Changing weight of locality[1] to 0 caused it to be removed. It's subconn
	// should also be removed.
	//
	// NOTE: this is because we handle locality with weight 0 same as the
	// locality doesn't exist. If this changes in the future, this removeSubConn
	// behavior will also change.
	scToRemove2 := <-cc.removeSubConnCh
	if !reflect.DeepEqual(scToRemove2, sc2) {
		t.Fatalf("RemoveSubConn, want %v, got %v", sc2, scToRemove2)
	}

	// Test pick with two subconns different locality weight.
	p6 := <-cc.newPickerCh
	// Locality-1 will be not be picked, and locality-2 will be picked.
	// Locality-2 contains sc3 and sc4. So expect sc3, sc4.
	want = []balancer.SubConn{sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p6)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

// The EDS balancer gets EDS resp with unhealthy endpoints. Test that only
// healthy ones are used.
func TestEDS_EndpointsHealth(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	// Two localities, each 3 backend, one Healthy, one Unhealthy, one Unknown.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:6], &xdsclient.AddLocalityOptions{
		Health: []corepb.HealthStatus{
			corepb.HealthStatus_HEALTHY,
			corepb.HealthStatus_UNHEALTHY,
			corepb.HealthStatus_UNKNOWN,
			corepb.HealthStatus_DRAINING,
			corepb.HealthStatus_TIMEOUT,
			corepb.HealthStatus_DEGRADED,
		},
	})
	clab1.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[6:12], &xdsclient.AddLocalityOptions{
		Health: []corepb.HealthStatus{
			corepb.HealthStatus_HEALTHY,
			corepb.HealthStatus_UNHEALTHY,
			corepb.HealthStatus_UNKNOWN,
			corepb.HealthStatus_DRAINING,
			corepb.HealthStatus_TIMEOUT,
			corepb.HealthStatus_DEGRADED,
		},
	})
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	var (
		readySCs           []balancer.SubConn
		newSubConnAddrStrs []string
	)
	for i := 0; i < 4; i++ {
		addr := <-cc.newSubConnAddrsCh
		newSubConnAddrStrs = append(newSubConnAddrStrs, addr[0].Addr)
		sc := <-cc.newSubConnCh
		edsb.HandleSubConnStateChange(sc, connectivity.Connecting)
		edsb.HandleSubConnStateChange(sc, connectivity.Ready)
		readySCs = append(readySCs, sc)
	}

	wantNewSubConnAddrStrs := []string{
		testEndpointAddrs[0],
		testEndpointAddrs[2],
		testEndpointAddrs[6],
		testEndpointAddrs[8],
	}
	sortStrTrans := cmp.Transformer("Sort", func(in []string) []string {
		out := append([]string(nil), in...) // Copy input to avoid mutating it.
		sort.Strings(out)
		return out
	})
	if !cmp.Equal(newSubConnAddrStrs, wantNewSubConnAddrStrs, sortStrTrans) {
		t.Fatalf("want newSubConn with address %v, got %v", wantNewSubConnAddrStrs, newSubConnAddrStrs)
	}

	// There should be exactly 4 new SubConns. Check to make sure there's no
	// more subconns being created.
	select {
	case <-cc.newSubConnCh:
		t.Fatalf("Got unexpected new subconn")
	case <-time.After(time.Microsecond * 100):
	}

	// Test roundrobin with the subconns.
	p1 := <-cc.newPickerCh
	want := readySCs
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

func TestClose(t *testing.T) {
	edsb := newEDSBalancerImpl(nil, nil)
	// This is what could happen when switching between fallback and eds. This
	// make sure it doesn't panic.
	edsb.Close()
}

func init() {
	balancer.Register(&testConstBalancerBuilder{})
}

var errTestConstPicker = fmt.Errorf("const picker error")

type testConstBalancerBuilder struct{}

func (*testConstBalancerBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	return &testConstBalancer{cc: cc}
}

func (*testConstBalancerBuilder) Name() string {
	return "test-const-balancer"
}

type testConstBalancer struct {
	cc balancer.ClientConn
}

func (tb *testConstBalancer) HandleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	tb.cc.UpdateState(balancer.State{ConnectivityState: connectivity.Ready, Picker: &testConstPicker{err: errTestConstPicker}})
}

func (tb *testConstBalancer) HandleResolvedAddrs(a []resolver.Address, err error) {
	if len(a) == 0 {
		return
	}
	tb.cc.NewSubConn(a, balancer.NewSubConnOptions{})
}

func (*testConstBalancer) Close() {
}

type testConstPicker struct {
	err error
	sc  balancer.SubConn
}

func (tcp *testConstPicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	if tcp.err != nil {
		return balancer.PickResult{}, tcp.err
	}
	return balancer.PickResult{SubConn: tcp.sc}, nil
}

// Create XDS balancer, and update sub-balancer before handling eds responses.
// Then switch between round-robin and test-const-balancer after handling first
// eds response.
func TestEDS_UpdateSubBalancerName(t *testing.T) {
	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, nil)

	t.Logf("update sub-balancer to test-const-balancer")
	edsb.HandleChildPolicy("test-const-balancer", nil)

	// Two localities, each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	clab1.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))

	for i := 0; i < 2; i++ {
		sc := <-cc.newSubConnCh
		edsb.HandleSubConnStateChange(sc, connectivity.Ready)
	}

	p0 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		_, err := p0.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(err, errTestConstPicker) {
			t.Fatalf("picker.Pick, got err %q, want err %q", err, errTestConstPicker)
		}
	}

	t.Logf("update sub-balancer to round-robin")
	edsb.HandleChildPolicy(roundrobin.Name, nil)

	for i := 0; i < 2; i++ {
		<-cc.removeSubConnCh
	}

	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)

	// Test roundrobin with two subconns.
	p1 := <-cc.newPickerCh
	want := []balancer.SubConn{sc1, sc2}
	if err := isRoundRobin(want, subConnFromPicker(p1)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}

	t.Logf("update sub-balancer to test-const-balancer")
	edsb.HandleChildPolicy("test-const-balancer", nil)

	for i := 0; i < 2; i++ {
		scToRemove := <-cc.removeSubConnCh
		if !reflect.DeepEqual(scToRemove, sc1) && !reflect.DeepEqual(scToRemove, sc2) {
			t.Fatalf("RemoveSubConn, want (%v or %v), got %v", sc1, sc2, scToRemove)
		}
		edsb.HandleSubConnStateChange(scToRemove, connectivity.Shutdown)
	}

	for i := 0; i < 2; i++ {
		sc := <-cc.newSubConnCh
		edsb.HandleSubConnStateChange(sc, connectivity.Ready)
	}

	p2 := <-cc.newPickerCh
	for i := 0; i < 5; i++ {
		_, err := p2.Pick(balancer.PickInfo{})
		if !reflect.DeepEqual(err, errTestConstPicker) {
			t.Fatalf("picker.Pick, got err %q, want err %q", err, errTestConstPicker)
		}
	}

	t.Logf("update sub-balancer to round-robin")
	edsb.HandleChildPolicy(roundrobin.Name, nil)

	for i := 0; i < 2; i++ {
		<-cc.removeSubConnCh
	}

	sc3 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc3, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc3, connectivity.Ready)
	sc4 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc4, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc4, connectivity.Ready)

	p3 := <-cc.newPickerCh
	want = []balancer.SubConn{sc3, sc4}
	if err := isRoundRobin(want, subConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

func TestDropPicker(t *testing.T) {
	const pickCount = 12
	var constPicker = &testConstPicker{
		sc: testSubConns[0],
	}

	tests := []struct {
		name  string
		drops []*dropper
	}{
		{
			name:  "no drop",
			drops: nil,
		},
		{
			name: "one drop",
			drops: []*dropper{
				newDropper(1, 2, ""),
			},
		},
		{
			name: "two drops",
			drops: []*dropper{
				newDropper(1, 3, ""),
				newDropper(1, 2, ""),
			},
		},
		{
			name: "three drops",
			drops: []*dropper{
				newDropper(1, 3, ""),
				newDropper(1, 4, ""),
				newDropper(1, 2, ""),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			p := newDropPicker(constPicker, tt.drops, nil)

			// scCount is the number of sc's returned by pick. The opposite of
			// drop-count.
			var (
				scCount   int
				wantCount = pickCount
			)
			for _, dp := range tt.drops {
				wantCount = wantCount * int(dp.denominator-dp.numerator) / int(dp.denominator)
			}

			for i := 0; i < pickCount; i++ {
				_, err := p.Pick(balancer.PickInfo{})
				if err == nil {
					scCount++
				}
			}

			if scCount != (wantCount) {
				t.Errorf("drops: %+v, scCount %v, wantCount %v", tt.drops, scCount, wantCount)
			}
		})
	}
}

func TestEDS_LoadReport(t *testing.T) {
	testLoadStore := newTestLoadStore()

	cc := newTestClientConn(t)
	edsb := newEDSBalancerImpl(cc, testLoadStore)

	backendToBalancerID := make(map[balancer.SubConn]internal.Locality)

	// Two localities, each with one backend.
	clab1 := xdsclient.NewClusterLoadAssignmentBuilder(testClusterNames[0], nil)
	clab1.AddLocality(testSubZones[0], 1, 0, testEndpointAddrs[:1], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))
	sc1 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc1, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc1, connectivity.Ready)
	backendToBalancerID[sc1] = internal.Locality{
		SubZone: testSubZones[0],
	}

	// Add the second locality later to make sure sc2 belongs to the second
	// locality. Otherwise the test is flaky because of a map is used in EDS to
	// keep localities.
	clab1.AddLocality(testSubZones[1], 1, 0, testEndpointAddrs[1:2], nil)
	edsb.HandleEDSResponse(xdsclient.ParseEDSRespProtoForTesting(clab1.Build()))
	sc2 := <-cc.newSubConnCh
	edsb.HandleSubConnStateChange(sc2, connectivity.Connecting)
	edsb.HandleSubConnStateChange(sc2, connectivity.Ready)
	backendToBalancerID[sc2] = internal.Locality{
		SubZone: testSubZones[1],
	}

	// Test roundrobin with two subconns.
	p1 := <-cc.newPickerCh
	var (
		wantStart []internal.Locality
		wantEnd   []internal.Locality
	)

	for i := 0; i < 10; i++ {
		scst, _ := p1.Pick(balancer.PickInfo{})
		locality := backendToBalancerID[scst.SubConn]
		wantStart = append(wantStart, locality)
		if scst.Done != nil && scst.SubConn != sc1 {
			scst.Done(balancer.DoneInfo{})
			wantEnd = append(wantEnd, backendToBalancerID[scst.SubConn])
		}
	}

	if !reflect.DeepEqual(testLoadStore.callsStarted, wantStart) {
		t.Fatalf("want started: %v, got: %v", testLoadStore.callsStarted, wantStart)
	}
	if !reflect.DeepEqual(testLoadStore.callsEnded, wantEnd) {
		t.Fatalf("want ended: %v, got: %v", testLoadStore.callsEnded, wantEnd)
	}
}
