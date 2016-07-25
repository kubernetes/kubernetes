// Copyright 2015 flannel authors
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

package subnet

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/jonboulle/clockwork"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
)

func newDummyRegistry() *MockSubnetRegistry {
	attrs := LeaseAttrs{
		PublicIP: ip.MustParseIP4("1.1.1.1"),
	}

	exp := time.Time{}

	subnets := []Lease{
		// leases within SubnetMin-SubnetMax range
		{ip.IP4Net{ip.MustParseIP4("10.3.1.0"), 24}, attrs, exp, 10},
		{ip.IP4Net{ip.MustParseIP4("10.3.2.0"), 24}, attrs, exp, 11},
		{ip.IP4Net{ip.MustParseIP4("10.3.4.0"), 24}, attrs, exp, 12},
		{ip.IP4Net{ip.MustParseIP4("10.3.5.0"), 24}, attrs, exp, 13},

		// hand created lease outside the range of subnetMin-SubnetMax for testing removal
		{ip.IP4Net{ip.MustParseIP4("10.3.31.0"), 24}, attrs, exp, 13},
	}

	config := `{ "Network": "10.3.0.0/16", "SubnetMin": "10.3.1.0", "SubnetMax": "10.3.25.0" }`
	return NewMockRegistry("_", config, subnets)
}

func TestAcquireLease(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP: extIaddr,
	}

	l, err := sm.AcquireLease(context.Background(), "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if !inAllocatableRange(context.Background(), sm, l.Subnet) {
		t.Fatal("Subnet mismatch: expected 10.3.3.0/24, got: ", l.Subnet)
	}

	// Acquire again, should reuse
	l2, err := sm.AcquireLease(context.Background(), "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if !l.Subnet.Equal(l2.Subnet) {
		t.Fatalf("AcquireLease did not reuse subnet; expected %v, got %v", l.Subnet, l2.Subnet)
	}
}

func TestConfigChanged(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP: extIaddr,
	}

	l, err := sm.AcquireLease(context.Background(), "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if !inAllocatableRange(context.Background(), sm, l.Subnet) {
		t.Fatal("Acquired subnet outside of valid range: ", l.Subnet)
	}

	// Change config
	config := `{ "Network": "10.4.0.0/16" }`
	msr.setConfig("_", config)

	// Acquire again, should not reuse
	if l, err = sm.AcquireLease(context.Background(), "_", &attrs); err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if !inAllocatableRange(context.Background(), sm, l.Subnet) {
		t.Fatal("Acquired subnet outside of valid range: ", l.Subnet)
	}
}

func newIP4Net(ipaddr string, prefix uint) ip.IP4Net {
	a, err := ip.ParseIP4(ipaddr)
	if err != nil {
		panic("failed to parse ipaddr")
	}
	return ip.IP4Net{
		IP:        a,
		PrefixLen: prefix,
	}
}

func acquireLease(ctx context.Context, t *testing.T, sm Manager) *Lease {
	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP: extIaddr,
	}

	l, err := sm.AcquireLease(ctx, "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	return l
}

func TestWatchLeaseAdded(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l := acquireLease(ctx, t, sm)

	events := make(chan []Event)
	go WatchLeases(ctx, sm, "_", l, events)

	evtBatch := <-events
	for _, evt := range evtBatch {
		if evt.Lease.Key() == l.Key() {
			t.Errorf("WatchLeases returned our own lease")
		}
	}

	expected := ip.IP4Net{
		IP:        ip.MustParseIP4("10.3.30.0"),
		PrefixLen: 24,
	}
	// Sanity check to make sure acquired lease is not this.
	// It shouldn't be as SubnetMin/SubnetMax in config is [10.3.1.0/24 to 10.3.25.0/24]
	if l.Subnet.Equal(expected) {
		t.Fatalf("Acquired lease conflicts with one about to create")
	}

	attrs := &LeaseAttrs{
		PublicIP: ip.MustParseIP4("1.1.1.1"),
	}
	_, err := msr.createSubnet(ctx, "_", expected, attrs, 0)
	if err != nil {
		t.Fatalf("createSubnet filed: %v", err)
	}

	evtBatch = <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchLeases produced wrong sized event batch: got %v, expected 1", len(evtBatch))
	}

	evt := evtBatch[0]

	if evt.Type != EventAdded {
		t.Fatalf("WatchLeases produced wrong event type")
	}

	actual := evt.Lease.Subnet
	if !actual.Equal(expected) {
		t.Errorf("WatchSubnet produced wrong subnet: expected %s, got %s", expected, actual)
	}
}

func TestWatchLeaseRemoved(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l := acquireLease(ctx, t, sm)

	events := make(chan []Event)
	go WatchLeases(ctx, sm, "_", l, events)

	evtBatch := <-events

	for _, evt := range evtBatch {
		if evt.Lease.Key() == l.Key() {
			t.Errorf("WatchLeases returned our own lease")
		}
	}

	expected := ip.IP4Net{ip.MustParseIP4("10.3.31.0"), 24}
	// Sanity check to make sure acquired lease is not this.
	// It shouldn't be as SubnetMin/SubnetMax in config is [10.3.1.0/24 to 10.3.25.0/24]
	if l.Subnet.Equal(expected) {
		t.Fatalf("Acquired lease conflicts with one about to create")
	}

	msr.expireSubnet("_", expected)

	evtBatch = <-events
	if len(evtBatch) != 1 {
		t.Fatalf("WatchLeases produced wrong sized event batch: %#v", evtBatch)
	}

	evt := evtBatch[0]

	if evt.Type != EventRemoved {
		t.Fatalf("WatchLeases produced wrong event type")
	}

	actual := evt.Lease.Subnet
	if !actual.Equal(expected) {
		t.Errorf("WatchSubnet produced wrong subnet: expected %s, got %s", expected, actual)
	}
}

type leaseData struct {
	Dummy string
}

func TestRenewLease(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)
	now := time.Now()
	fakeClock := clockwork.NewFakeClockAt(now)
	clock = fakeClock

	// Create LeaseAttrs
	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP:    extIaddr,
		BackendType: "vxlan",
	}

	ld, err := json.Marshal(&leaseData{Dummy: "test string"})
	if err != nil {
		t.Fatalf("Failed to marshal leaseData: %v", err)
	}
	attrs.BackendData = json.RawMessage(ld)

	// Acquire lease
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l, err := sm.AcquireLease(ctx, "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	now = now.Add(subnetTTL)

	fakeClock.Advance(24 * time.Hour)

	if err := sm.RenewLease(ctx, "_", l); err != nil {
		t.Fatal("RenewLease failed: ", err)
	}

	// check that it's still good
	n, err := msr.getNetwork(ctx, "_")
	if err != nil {
		t.Error("Failed to renew lease: could not get networks: %v", err)
	}

	for _, sn := range n.subnets {
		if sn.Subnet.Equal(l.Subnet) {
			expected := now.Add(subnetTTL)
			if !sn.Expiration.Equal(expected) {
				t.Errorf("Failed to renew lease: bad expiration; expected %v, got %v", expected, sn.Expiration)
			}
			if !reflect.DeepEqual(sn.Attrs, attrs) {
				t.Errorf("LeaseAttrs changed: was %#v, now %#v", attrs, sn.Attrs)
			}
			return
		}
	}

	t.Fatalf("Failed to find acquired lease")
}

func TestLeaseRevoked(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l := acquireLease(ctx, t, sm)

	if err := sm.RevokeLease(ctx, "_", l.Subnet); err != nil {
		t.Fatalf("RevokeLease failed: %v", err)
	}

	_, _, err := msr.getSubnet(ctx, "_", l.Subnet)
	if err == nil {
		t.Fatalf("Revoked lease still exists")
	}
	if etcdErr, ok := err.(etcd.Error); ok && etcdErr.Code != etcd.ErrorCodeKeyNotFound {
		t.Fatalf("getSubnets after revoked lease returned unexpected error: %v", err)
	}
}

func TestWatchGetNetworks(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Kill the previously added "_" network
	msr.DeleteNetwork(ctx, "_")

	expected := "foobar"
	msr.CreateNetwork(ctx, expected, `{"Network": "10.1.1.0/16", "Backend": {"Type": "bridge"}}`)

	resp, err := sm.WatchNetworks(ctx, nil)
	if err != nil {
		t.Errorf("WatchNetworks(nil) failed:", err)
	}

	if len(resp.Snapshot) != 1 {
		t.Errorf("WatchNetworks(nil) produced wrong number of networks: expected 1, got %d", len(resp.Snapshot))
	}

	if resp.Snapshot[0] != expected {
		t.Errorf("WatchNetworks(nil) produced wrong network: expected %s, got %s", expected, resp.Snapshot[0])
	}
}

func TestWatchNetworkAdded(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events := make(chan []Event)
	go WatchNetworks(ctx, sm, events)

	// skip over the initial snapshot
	<-events

	expected := "foobar"
	msr.CreateNetwork(ctx, expected, `{"Network": "10.1.1.0/16", "Backend": {"Type": "bridge"}}`)

	evtBatch := <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchNetworks produced wrong sized event batch")
	}

	evt := evtBatch[0]

	if evt.Type != EventAdded {
		t.Fatalf("WatchNetworks produced wrong event type")
	}

	actual := evt.Network
	if actual != expected {
		t.Errorf("WatchNetworks produced wrong network: expected %s, got %s", expected, actual)
	}
}

func TestWatchNetworkRemoved(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events := make(chan []Event)
	go WatchNetworks(ctx, sm, events)

	// skip over the initial snapshot
	<-events

	expected := "blah"
	msr.CreateNetwork(ctx, expected, `{"Network": "10.1.1.0/16", "Backend": {"Type": "bridge"}}`)

	// skip over the create event
	<-events

	err := msr.DeleteNetwork(ctx, expected)
	if err != nil {
		t.Fatalf("WatchNetworks failed to delete the network")
	}

	evtBatch := <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchNetworks produced wrong sized event batch")
	}

	evt := evtBatch[0]

	if evt.Type != EventRemoved {
		t.Fatalf("WatchNetworks produced wrong event type")
	}

	actual := evt.Network
	if actual != expected {
		t.Errorf("WatchNetwork produced wrong network: expected %s, got %s", expected, actual)
	}
}

func TestAddReservation(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r := Reservation{
		Subnet:   newIP4Net("10.4.3.0", 24),
		PublicIP: ip.MustParseIP4("52.195.12.13"),
	}
	if err := sm.AddReservation(ctx, "_", &r); err == nil {
		t.Fatalf("unexpectedly added a reservation outside of configured network")
	}

	r.Subnet = newIP4Net("10.3.10.0", 24)
	if err := sm.AddReservation(ctx, "_", &r); err != nil {
		t.Fatalf("failed to add reservation: %v", err)
	}

	// Add the same reservation -- should succeed
	if err := sm.AddReservation(ctx, "_", &r); err != nil {
		t.Fatalf("failed to add reservation: %v", err)
	}

	// Add a reservation with a different public IP -- should fail
	r2 := r
	r2.PublicIP = ip.MustParseIP4("52.195.12.17")
	if err := sm.AddReservation(ctx, "_", &r2); err != ErrLeaseTaken {
		t.Fatalf("taken add reservation returned: %v", err)
	}

	attrs := &LeaseAttrs{
		PublicIP: r.PublicIP,
	}
	l, err := sm.AcquireLease(ctx, "_", attrs)
	if err != nil {
		t.Fatalf("failed to acquire subnet: %v", err)
	}
	if !l.Subnet.Equal(r.Subnet) {
		t.Fatalf("acquired subnet is not the reserved one: expected %v, got %v", r.Subnet, l.Subnet)
	}
	if !l.Expiration.IsZero() {
		t.Fatalf("acquired lease (prev reserved) has expiration set")
	}
}

func TestRemoveReservation(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r := Reservation{
		Subnet:   newIP4Net("10.3.10.0", 24),
		PublicIP: ip.MustParseIP4("52.195.12.13"),
	}
	if err := sm.AddReservation(ctx, "_", &r); err != nil {
		t.Fatalf("failed to add reservation: %v", err)
	}

	if err := sm.RemoveReservation(ctx, "_", r.Subnet); err != nil {
		t.Fatalf("failed to remove reservation: %v", err)
	}

	// The node should have a TTL
	sub, _, err := msr.getSubnet(ctx, "_", r.Subnet)
	if err != nil {
		t.Fatalf("getSubnet failed: %v", err)
	}

	if sub.Expiration.IsZero() {
		t.Fatalf("removed reservation resulted in no TTL")
	}
}

func TestListReservations(t *testing.T) {
	msr := newDummyRegistry()
	sm := NewMockManager(msr)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r1 := Reservation{
		Subnet:   newIP4Net("10.3.10.0", 24),
		PublicIP: ip.MustParseIP4("52.195.12.13"),
	}
	if err := sm.AddReservation(ctx, "_", &r1); err != nil {
		t.Fatalf("failed to add reservation: %v", err)
	}

	r2 := Reservation{
		Subnet:   newIP4Net("10.3.20.0", 24),
		PublicIP: ip.MustParseIP4("52.195.12.14"),
	}
	if err := sm.AddReservation(ctx, "_", &r2); err != nil {
		t.Fatalf("failed to add reservation: %v", err)
	}

	rs, err := sm.ListReservations(ctx, "_")
	if err != nil {
		if len(rs) != 2 {
			t.Fatalf("unexpected number of reservations, expected 2, got %v", len(rs))
		}
		if !resvEqual(rs[0], r1) && !resvEqual(rs[1], r1) {
			t.Fatalf("reservation not found")
		}
		if !resvEqual(rs[0], r2) && !resvEqual(rs[1], r2) {
			t.Fatalf("reservation not found")
		}
	}
}

func inAllocatableRange(ctx context.Context, sm Manager, ipn ip.IP4Net) bool {
	cfg, err := sm.GetNetworkConfig(ctx, "_")
	if err != nil {
		panic(err)
	}

	return ipn.IP >= cfg.SubnetMin || ipn.IP <= cfg.SubnetMax
}

func resvEqual(r1, r2 Reservation) bool {
	return r1.Subnet.Equal(r2.Subnet) && r1.PublicIP == r2.PublicIP
}
