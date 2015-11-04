// Copyright 2015 CoreOS, Inc.
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
	"fmt"
	"reflect"
	"testing"
	"time"

	etcd "github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/client"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
)

func newDummyRegistry(ttlOverride time.Duration) *MockSubnetRegistry {
	subnets := []*etcd.Node{
		&etcd.Node{Key: "10.3.1.0-24", Value: `{ "PublicIP": "1.1.1.1" }`, ModifiedIndex: 10},
		&etcd.Node{Key: "10.3.2.0-24", Value: `{ "PublicIP": "1.1.1.1" }`, ModifiedIndex: 11},
		&etcd.Node{Key: "10.3.4.0-24", Value: `{ "PublicIP": "1.1.1.1" }`, ModifiedIndex: 12},
		&etcd.Node{Key: "10.3.5.0-24", Value: `{ "PublicIP": "1.1.1.1" }`, ModifiedIndex: 13},
	}

	config := `{ "Network": "10.3.0.0/16", "SubnetMin": "10.3.1.0", "SubnetMax": "10.3.5.0" }`
	return NewMockRegistry(ttlOverride, "_", config, subnets)
}

func TestAcquireLease(t *testing.T) {
	msr := newDummyRegistry(1000)
	sm := newEtcdManager(msr)

	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP: extIaddr,
	}

	l, err := sm.AcquireLease(context.Background(), "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if l.Subnet.String() != "10.3.3.0/24" {
		t.Fatal("Subnet mismatch: expected 10.3.3.0/24, got: ", l.Subnet)
	}

	// Acquire again, should reuse
	if l, err = sm.AcquireLease(context.Background(), "_", &attrs); err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if l.Subnet.String() != "10.3.3.0/24" {
		t.Fatal("Subnet mismatch: expected 10.3.3.0/24, got: ", l.Subnet)
	}
}

func TestConfigChanged(t *testing.T) {
	msr := newDummyRegistry(1000)
	sm := newEtcdManager(msr)

	extIaddr, _ := ip.ParseIP4("1.2.3.4")
	attrs := LeaseAttrs{
		PublicIP: extIaddr,
	}

	l, err := sm.AcquireLease(context.Background(), "_", &attrs)
	if err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	if l.Subnet.String() != "10.3.3.0/24" {
		t.Fatal("Subnet mismatch: expected 10.3.3.0/24, got: ", l.Subnet)
	}

	// Change config
	config := `{ "Network": "10.4.0.0/16" }`
	msr.setConfig("_", config)

	// Acquire again, should not reuse
	if l, err = sm.AcquireLease(context.Background(), "_", &attrs); err != nil {
		t.Fatal("AcquireLease failed: ", err)
	}

	newNet := newIP4Net("10.4.0.0", 16)
	if !newNet.Contains(l.Subnet.IP) {
		t.Fatalf("Subnet mismatch: expected within %v, got: %v", newNet, l.Subnet)
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
	msr := newDummyRegistry(0)
	sm := newEtcdManager(msr)

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

	expected := "10.3.6.0-24"
	msr.createSubnet(ctx, "_", expected, `{"PublicIP": "1.1.1.1"}`, 0)

	evtBatch = <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchLeases produced wrong sized event batch")
	}

	evt := evtBatch[0]

	if evt.Type != EventAdded {
		t.Fatalf("WatchLeases produced wrong event type")
	}

	actual := evt.Lease.Key()
	if actual != expected {
		t.Errorf("WatchSubnet produced wrong subnet: expected %s, got %s", expected, actual)
	}
}

func TestWatchLeaseRemoved(t *testing.T) {
	msr := newDummyRegistry(0)
	sm := newEtcdManager(msr)

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

	expected := "10.3.4.0-24"
	msr.expireSubnet("_", expected)

	evtBatch = <-events
	if len(evtBatch) != 1 {
		t.Fatalf("WatchLeases produced wrong sized event batch")
	}

	evt := evtBatch[0]

	if evt.Type != EventRemoved {
		t.Fatalf("WatchLeases produced wrong event type")
	}

	actual := evt.Lease.Key()
	if actual != expected {
		t.Errorf("WatchSubnet produced wrong subnet: expected %s, got %s", expected, actual)
	}
}

type leaseData struct {
	Dummy string
}

func TestRenewLease(t *testing.T) {
	msr := newDummyRegistry(1)
	sm := newEtcdManager(msr)

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

	go LeaseRenewer(ctx, sm, "_", l)

	fmt.Println("Waiting for lease to pass original expiration")
	time.Sleep(2 * time.Second)

	// check that it's still good
	net, err := msr.getNetwork(ctx, "_")
	if err != nil {
		t.Error("Failed to renew lease: could not get networks: %v", err)
	}
	for _, n := range net.Node.Nodes {
		if n.Key == l.Subnet.StringSep(".", "-") {
			if n.Expiration.Before(time.Now()) {
				t.Error("Failed to renew lease: expiration did not advance")
			}
			a := LeaseAttrs{}
			if err := json.Unmarshal([]byte(n.Value), &a); err != nil {
				t.Errorf("Failed to JSON-decode LeaseAttrs: %v", err)
				return
			}
			if !reflect.DeepEqual(a, attrs) {
				t.Errorf("LeaseAttrs changed: was %#v, now %#v", attrs, a)
			}
			return
		}
	}

	t.Fatalf("Failed to find acquired lease")
}

func TestWatchGetNetworks(t *testing.T) {
	msr := newDummyRegistry(0)
	sm := newEtcdManager(msr)

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
	msr := newDummyRegistry(0)
	sm := newEtcdManager(msr)

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
	msr := newDummyRegistry(0)
	sm := newEtcdManager(msr)

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

	_, err := msr.DeleteNetwork(ctx, expected)
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
