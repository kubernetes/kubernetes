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
	"fmt"
	"sync"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
)

func newTestEtcdRegistry(t *testing.T) (Registry, *mockEtcd) {
	cfg := &EtcdConfig{
		Endpoints: []string{"http://127.0.0.1:4001", "http://127.0.0.1:2379"},
		Prefix:    "/coreos.com/network",
	}

	r, err := newEtcdSubnetRegistry(cfg, func(c *EtcdConfig) (etcd.KeysAPI, error) {
		return newMockEtcd(), nil
	})
	if err != nil {
		t.Fatal("Failed to create etcd subnet registry")
	}

	return r, r.(*etcdSubnetRegistry).cli.(*mockEtcd)
}

func watchSubnets(t *testing.T, r Registry, ctx context.Context, sn ip.IP4Net, nextIndex uint64, result chan error) {
	type leaseEvent struct {
		etype  EventType
		subnet ip.IP4Net
		found  bool
	}
	expectedEvents := []leaseEvent{
		{EventAdded, sn, false},
		{EventRemoved, sn, false},
	}

	numFound := 0
	for {
		evt, index, err := r.watchSubnets(ctx, "foobar", nextIndex)

		switch {
		case err == nil:
			nextIndex = index
			for _, exp := range expectedEvents {
				if evt.Type != exp.etype {
					continue
				}
				if exp.found == true {
					result <- fmt.Errorf("Subnet event type already found: %v", exp)
					return
				}
				if !evt.Lease.Subnet.Equal(exp.subnet) {
					result <- fmt.Errorf("Subnet event lease %v mismatch (expected %v)", evt.Lease.Subnet, exp.subnet)
				}
				exp.found = true
				numFound += 1
			}
			if numFound == len(expectedEvents) {
				// All done; success
				result <- nil
				return
			}
		case isIndexTooSmall(err):
			nextIndex = err.(etcd.Error).Index

		default:
			result <- fmt.Errorf("Error watching subnet leases: %v", err)
			return
		}
	}

	result <- fmt.Errorf("Should never get here")
}

func TestEtcdRegistry(t *testing.T) {
	r, m := newTestEtcdRegistry(t)

	ctx, _ := context.WithCancel(context.Background())

	networks, _, err := r.getNetworks(ctx)
	if err != nil {
		t.Fatal("Failed to get networks")
	}
	if len(networks) != 0 {
		t.Fatal("Networks should be empty")
	}

	// Populate etcd with a network
	netKey := "/coreos.com/network/foobar/config"
	netValue := "{ \"Network\": \"10.1.0.0/16\", \"Backend\": { \"Type\": \"host-gw\" } }"
	m.Create(ctx, netKey, netValue)

	networks, _, err = r.getNetworks(ctx)
	if err != nil {
		t.Fatal("Failed to get networks the second time")
	}
	if len(networks) != 1 {
		t.Fatal("Failed to find expected network foobar")
	}

	config, err := r.getNetworkConfig(ctx, "foobar")
	if err != nil {
		t.Fatal("Failed to get network config")
	}
	if config != netValue {
		t.Fatal("Failed to match network config")
	}

	sn := ip.IP4Net{
		IP:        ip.MustParseIP4("10.1.5.0"),
		PrefixLen: 24,
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	startWg := sync.WaitGroup{}
	startWg.Add(1)
	result := make(chan error, 1)
	go func() {
		startWg.Done()
		watchSubnets(t, r, ctx, sn, m.index, result)
		wg.Done()
	}()

	startWg.Wait()
	// Lease a subnet for the network
	attrs := &LeaseAttrs{
		PublicIP: ip.MustParseIP4("1.2.3.4"),
	}
	exp, err := r.createSubnet(ctx, "foobar", sn, attrs, 24*time.Hour)
	if err != nil {
		t.Fatal("Failed to create subnet lease")
	}
	if !exp.After(time.Now()) {
		t.Fatal("Subnet lease duration %v not in the future", exp)
	}

	// Make sure the lease got created
	resp, err := m.Get(ctx, "/coreos.com/network/foobar/subnets/10.1.5.0-24", nil)
	if err != nil {
		t.Fatal("Failed to verify subnet lease directly in etcd: %v", err)
	}
	if resp == nil || resp.Node == nil {
		t.Fatal("Failed to retrive node in subnet lease")
	}
	if resp.Node.Value != "{\"PublicIP\":\"1.2.3.4\"}" {
		t.Fatal("Unexpected subnet lease node %s value %s", resp.Node.Key, resp.Node.Value)
	}

	leases, _, err := r.getSubnets(ctx, "foobar")
	if len(leases) != 1 {
		t.Fatalf("Unexpected number of leases %d (expected 1)", len(leases))
	}
	if !leases[0].Subnet.Equal(sn) {
		t.Fatalf("Mismatched subnet %v (expected %v)", leases[0].Subnet, sn)
	}

	lease, _, err := r.getSubnet(ctx, "foobar", sn)
	if lease == nil {
		t.Fatal("Missing subnet lease")
	}

	err = r.deleteSubnet(ctx, "foobar", sn)
	if err != nil {
		t.Fatalf("Failed to delete subnet %v: %v", sn, err)
	}

	// Make sure the lease got deleted
	resp, err = m.Get(ctx, "/coreos.com/network/foobar/subnets/10.1.5.0-24", nil)
	if err == nil {
		t.Fatal("Unexpected success getting deleted subnet")
	}

	wg.Wait()

	// Check errors from watch goroutine
	watchResult := <-result
	if watchResult != nil {
		t.Fatalf("Error watching keys: %v", watchResult)
	}

	// TODO: watchSubnet and watchNetworks
}
