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

package remote

import (
	"fmt"
	"net"
	"net/url"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

const expectedNetwork = "10.1.0.0/16"

func TestRemote(t *testing.T) {
	config := fmt.Sprintf(`{"Network": %q}`, expectedNetwork)
	serverRegistry := subnet.NewMockRegistry(1, "", config, nil)
	sm := subnet.NewMockManager(serverRegistry)

	addr := "127.0.0.1:9999"

	ctx, cancel := context.WithCancel(context.Background())
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		RunServer(ctx, sm, addr, "", "", "")
		wg.Done()
	}()

	doTestRemote(ctx, t, addr, serverRegistry)

	cancel()
	wg.Wait()
}

func mustParseIP4(s string) ip.IP4 {
	a, err := ip.ParseIP4(s)
	if err != nil {
		panic(err)
	}
	return a
}

func mustParseIP4Net(s string) ip.IP4Net {
	_, n, err := net.ParseCIDR(s)
	if err != nil {
		panic(err)
	}
	return ip.FromIPNet(n)
}

func isConnRefused(err error) bool {
	if uerr, ok := err.(*url.Error); ok {
		if operr, ok := uerr.Err.(*net.OpError); ok {
			return operr.Err == syscall.ECONNREFUSED
		}
	}
	return false
}

func doTestRemote(ctx context.Context, t *testing.T, remoteAddr string, serverRegistry *subnet.MockSubnetRegistry) {
	sm, err := NewRemoteManager(remoteAddr, "", "", "")
	if err != nil {
		t.Fatalf("Failed to create remote mananager: %v", err)
	}

	for i := 0; ; i++ {
		cfg, err := sm.GetNetworkConfig(ctx, "_")
		if err != nil {
			if isConnRefused(err) {
				if i == 100 {
					t.Fatalf("Out of connection retries")
				}

				fmt.Println("Connection refused, retrying...")
				time.Sleep(300 * time.Millisecond)
				continue
			}

			t.Fatalf("GetNetworkConfig failed: %v", err)
		}

		if cfg.Network.String() != expectedNetwork {
			t.Errorf("GetNetworkConfig returned bad network: %v vs %v", cfg.Network, expectedNetwork)
		}
		break
	}

	attrs := &subnet.LeaseAttrs{
		PublicIP: mustParseIP4("1.1.1.1"),
	}
	l, err := sm.AcquireLease(ctx, "_", attrs)
	if err != nil {
		t.Fatalf("AcquireLease failed: %v", err)
	}

	if !mustParseIP4Net(expectedNetwork).Contains(l.Subnet.IP) {
		t.Errorf("AcquireLease returned subnet not in network: %v (in %v)", l.Subnet, expectedNetwork)
	}

	if err = sm.RenewLease(ctx, "_", l); err != nil {
		t.Errorf("RenewLease failed: %v", err)
	}

	doTestWatchLeases(t, sm)

	doTestWatchNetworks(t, sm, serverRegistry)
}

func doTestWatchLeases(t *testing.T, sm subnet.Manager) {
	ctx, cancel := context.WithCancel(context.Background())
	wg := sync.WaitGroup{}
	wg.Add(1)
	defer func() {
		cancel()
		wg.Wait()
	}()

	events := make(chan []subnet.Event)
	go func() {
		subnet.WatchLeases(ctx, sm, "_", nil, events)
		wg.Done()
	}()

	// skip over the initial snapshot
	<-events

	attrs := &subnet.LeaseAttrs{
		PublicIP: mustParseIP4("1.1.1.2"),
	}
	l, err := sm.AcquireLease(ctx, "_", attrs)
	if err != nil {
		t.Errorf("AcquireLease failed: %v", err)
		return
	}
	if !mustParseIP4Net(expectedNetwork).Contains(l.Subnet.IP) {
		t.Errorf("AcquireLease returned subnet not in network: %v (in %v)", l.Subnet, expectedNetwork)
	}

	evtBatch := <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchSubnets produced wrong sized event batch")
	}

	evt := evtBatch[0]
	if evt.Type != subnet.EventAdded {
		t.Fatalf("WatchSubnets produced wrong event type")
	}

	if evt.Lease.Key() != l.Key() {
		t.Errorf("WatchSubnet produced wrong subnet: expected %s, got %s", l.Key(), evt.Lease.Key())
	}
}

func doTestWatchNetworks(t *testing.T, sm subnet.Manager, serverRegistry *subnet.MockSubnetRegistry) {
	ctx, cancel := context.WithCancel(context.Background())
	wg := sync.WaitGroup{}
	wg.Add(1)
	defer func() {
		cancel()
		wg.Wait()
	}()

	events := make(chan []subnet.Event)
	go func() {
		subnet.WatchNetworks(ctx, sm, events)
		wg.Done()
	}()

	// skip over the initial snapshot
	<-events

	expectedNetname := "foobar"
	config := fmt.Sprintf(`{"Network": %q}`, expectedNetwork)
	_, err := serverRegistry.CreateNetwork(ctx, expectedNetname, config)
	if err != nil {
		t.Errorf("create network failed: %v", err)
	}

	evtBatch := <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchNetworks create produced wrong sized event batch")
	}

	evt := evtBatch[0]
	if evt.Type != subnet.EventAdded {
		t.Fatalf("WatchNetworks create produced wrong event type")
	}

	if evt.Network != expectedNetname {
		t.Errorf("WatchNetwork create produced wrong network: expected %s, got %s", expectedNetname, evt.Network)
	}

	_, err = serverRegistry.DeleteNetwork(ctx, expectedNetname)
	if err != nil {
		t.Errorf("delete network failed: %v", err)
	}

	evtBatch = <-events

	if len(evtBatch) != 1 {
		t.Fatalf("WatchNetworks delete produced wrong sized event batch")
	}

	evt = evtBatch[0]
	if evt.Type != subnet.EventRemoved {
		t.Fatalf("WatchNetworks delete produced wrong event type")
	}

	if evt.Network != expectedNetname {
		t.Errorf("WatchNetwork delete produced wrong network: expected %s, got %s", expectedNetname, evt.Network)
	}
}
