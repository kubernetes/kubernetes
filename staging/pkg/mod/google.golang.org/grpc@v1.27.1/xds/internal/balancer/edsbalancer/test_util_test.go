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
	"context"
	"fmt"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/xds/internal"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
)

const testSubConnsCount = 16

var testSubConns []*testSubConn

func init() {
	for i := 0; i < testSubConnsCount; i++ {
		testSubConns = append(testSubConns, &testSubConn{
			id: fmt.Sprintf("sc%d", i),
		})
	}
}

type testSubConn struct {
	id string
}

func (tsc *testSubConn) UpdateAddresses([]resolver.Address) {
	panic("not implemented")
}

func (tsc *testSubConn) Connect() {
}

// Implement stringer to get human friendly error message.
func (tsc *testSubConn) String() string {
	return tsc.id
}

type testClientConn struct {
	t *testing.T // For logging only.

	newSubConnAddrsCh chan []resolver.Address // The last 10 []Address to create subconn.
	newSubConnCh      chan balancer.SubConn   // The last 10 subconn created.
	removeSubConnCh   chan balancer.SubConn   // The last 10 subconn removed.

	newPickerCh chan balancer.V2Picker  // The last picker updated.
	newStateCh  chan connectivity.State // The last state.

	subConnIdx int
}

func newTestClientConn(t *testing.T) *testClientConn {
	return &testClientConn{
		t: t,

		newSubConnAddrsCh: make(chan []resolver.Address, 10),
		newSubConnCh:      make(chan balancer.SubConn, 10),
		removeSubConnCh:   make(chan balancer.SubConn, 10),

		newPickerCh: make(chan balancer.V2Picker, 1),
		newStateCh:  make(chan connectivity.State, 1),
	}
}

func (tcc *testClientConn) NewSubConn(a []resolver.Address, o balancer.NewSubConnOptions) (balancer.SubConn, error) {
	sc := testSubConns[tcc.subConnIdx]
	tcc.subConnIdx++

	tcc.t.Logf("testClientConn: NewSubConn(%v, %+v) => %s", a, o, sc)
	select {
	case tcc.newSubConnAddrsCh <- a:
	default:
	}

	select {
	case tcc.newSubConnCh <- sc:
	default:
	}

	return sc, nil
}

func (tcc *testClientConn) RemoveSubConn(sc balancer.SubConn) {
	tcc.t.Logf("testClientCOnn: RemoveSubConn(%p)", sc)
	select {
	case tcc.removeSubConnCh <- sc:
	default:
	}
}

func (tcc *testClientConn) UpdateBalancerState(s connectivity.State, p balancer.Picker) {
	tcc.t.Fatal("not implemented")
}

func (tcc *testClientConn) UpdateState(bs balancer.State) {
	tcc.t.Logf("testClientConn: UpdateState(%v)", bs)
	select {
	case <-tcc.newStateCh:
	default:
	}
	tcc.newStateCh <- bs.ConnectivityState

	select {
	case <-tcc.newPickerCh:
	default:
	}
	tcc.newPickerCh <- bs.Picker
}

func (tcc *testClientConn) ResolveNow(resolver.ResolveNowOptions) {
	panic("not implemented")
}

func (tcc *testClientConn) Target() string {
	panic("not implemented")
}

type testServerLoad struct {
	name string
	d    float64
}

type testLoadStore struct {
	callsStarted []internal.Locality
	callsEnded   []internal.Locality
	callsCost    []testServerLoad
}

func newTestLoadStore() *testLoadStore {
	return &testLoadStore{}
}

func (*testLoadStore) CallDropped(category string) {
	panic("not implemented")
}

func (tls *testLoadStore) CallStarted(l internal.Locality) {
	tls.callsStarted = append(tls.callsStarted, l)
}

func (tls *testLoadStore) CallFinished(l internal.Locality, err error) {
	tls.callsEnded = append(tls.callsEnded, l)
}

func (tls *testLoadStore) CallServerLoad(l internal.Locality, name string, d float64) {
	tls.callsCost = append(tls.callsCost, testServerLoad{name: name, d: d})
}

func (*testLoadStore) ReportTo(ctx context.Context, cc *grpc.ClientConn, clusterName string, node *corepb.Node) {
	panic("not implemented")
}

// isRoundRobin checks whether f's return value is roundrobin of elements from
// want. But it doesn't check for the order. Note that want can contain
// duplicate items, which makes it weight-round-robin.
//
// Step 1. the return values of f should form a permutation of all elements in
// want, but not necessary in the same order. E.g. if want is {a,a,b}, the check
// fails if f returns:
//  - {a,a,a}: third a is returned before b
//  - {a,b,b}: second b is returned before the second a
//
// If error is found in this step, the returned error contains only the first
// iteration until where it goes wrong.
//
// Step 2. the return values of f should be repetitions of the same permutation.
// E.g. if want is {a,a,b}, the check failes if f returns:
//  - {a,b,a,b,a,a}: though it satisfies step 1, the second iteration is not
//  repeating the first iteration.
//
// If error is found in this step, the returned error contains the first
// iteration + the second iteration until where it goes wrong.
func isRoundRobin(want []balancer.SubConn, f func() balancer.SubConn) error {
	wantSet := make(map[balancer.SubConn]int) // SubConn -> count, for weighted RR.
	for _, sc := range want {
		wantSet[sc]++
	}

	// The first iteration: makes sure f's return values form a permutation of
	// elements in want.
	//
	// Also keep the returns values in a slice, so we can compare the order in
	// the second iteration.
	gotSliceFirstIteration := make([]balancer.SubConn, 0, len(want))
	for range want {
		got := f()
		gotSliceFirstIteration = append(gotSliceFirstIteration, got)
		wantSet[got]--
		if wantSet[got] < 0 {
			return fmt.Errorf("non-roundrobin want: %v, result: %v", want, gotSliceFirstIteration)
		}
	}

	// The second iteration should repeat the first iteration.
	var gotSliceSecondIteration []balancer.SubConn
	for i := 0; i < 2; i++ {
		for _, w := range gotSliceFirstIteration {
			g := f()
			gotSliceSecondIteration = append(gotSliceSecondIteration, g)
			if w != g {
				return fmt.Errorf("non-roundrobin, first iter: %v, second iter: %v", gotSliceFirstIteration, gotSliceSecondIteration)
			}
		}
	}

	return nil
}

// testClosure is a test util for TestIsRoundRobin.
type testClosure struct {
	r []balancer.SubConn
	i int
}

func (tc *testClosure) next() balancer.SubConn {
	ret := tc.r[tc.i]
	tc.i = (tc.i + 1) % len(tc.r)
	return ret
}

func TestIsRoundRobin(t *testing.T) {
	var (
		sc1 = testSubConns[0]
		sc2 = testSubConns[1]
		sc3 = testSubConns[2]
	)

	testCases := []struct {
		desc string
		want []balancer.SubConn
		got  []balancer.SubConn
		pass bool
	}{
		{
			desc: "0 element",
			want: []balancer.SubConn{},
			got:  []balancer.SubConn{},
			pass: true,
		},
		{
			desc: "1 element RR",
			want: []balancer.SubConn{sc1},
			got:  []balancer.SubConn{sc1, sc1, sc1, sc1},
			pass: true,
		},
		{
			desc: "1 element not RR",
			want: []balancer.SubConn{sc1},
			got:  []balancer.SubConn{sc1, sc2, sc1},
			pass: false,
		},
		{
			desc: "2 elements RR",
			want: []balancer.SubConn{sc1, sc2},
			got:  []balancer.SubConn{sc1, sc2, sc1, sc2, sc1, sc2},
			pass: true,
		},
		{
			desc: "2 elements RR different order from want",
			want: []balancer.SubConn{sc2, sc1},
			got:  []balancer.SubConn{sc1, sc2, sc1, sc2, sc1, sc2},
			pass: true,
		},
		{
			desc: "2 elements RR not RR, mistake in first iter",
			want: []balancer.SubConn{sc1, sc2},
			got:  []balancer.SubConn{sc1, sc1, sc1, sc2, sc1, sc2},
			pass: false,
		},
		{
			desc: "2 elements RR not RR, mistake in second iter",
			want: []balancer.SubConn{sc1, sc2},
			got:  []balancer.SubConn{sc1, sc2, sc1, sc1, sc1, sc2},
			pass: false,
		},
		{
			desc: "2 elements weighted RR",
			want: []balancer.SubConn{sc1, sc1, sc2},
			got:  []balancer.SubConn{sc1, sc1, sc2, sc1, sc1, sc2},
			pass: true,
		},
		{
			desc: "2 elements weighted RR different order",
			want: []balancer.SubConn{sc1, sc1, sc2},
			got:  []balancer.SubConn{sc1, sc2, sc1, sc1, sc2, sc1},
			pass: true,
		},

		{
			desc: "3 elements RR",
			want: []balancer.SubConn{sc1, sc2, sc3},
			got:  []balancer.SubConn{sc1, sc2, sc3, sc1, sc2, sc3, sc1, sc2, sc3},
			pass: true,
		},
		{
			desc: "3 elements RR different order",
			want: []balancer.SubConn{sc1, sc2, sc3},
			got:  []balancer.SubConn{sc3, sc2, sc1, sc3, sc2, sc1},
			pass: true,
		},
		{
			desc: "3 elements weighted RR",
			want: []balancer.SubConn{sc1, sc1, sc1, sc2, sc2, sc3},
			got:  []balancer.SubConn{sc1, sc2, sc3, sc1, sc2, sc1, sc1, sc2, sc3, sc1, sc2, sc1},
			pass: true,
		},
		{
			desc: "3 elements weighted RR not RR, mistake in first iter",
			want: []balancer.SubConn{sc1, sc1, sc1, sc2, sc2, sc3},
			got:  []balancer.SubConn{sc1, sc2, sc1, sc1, sc2, sc1, sc1, sc2, sc3, sc1, sc2, sc1},
			pass: false,
		},
		{
			desc: "3 elements weighted RR not RR, mistake in second iter",
			want: []balancer.SubConn{sc1, sc1, sc1, sc2, sc2, sc3},
			got:  []balancer.SubConn{sc1, sc2, sc3, sc1, sc2, sc1, sc1, sc1, sc3, sc1, sc2, sc1},
			pass: false,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			err := isRoundRobin(tC.want, (&testClosure{r: tC.got}).next)
			if err == nil != tC.pass {
				t.Errorf("want pass %v, want %v, got err %v", tC.pass, tC.want, err)
			}
		})
	}
}
