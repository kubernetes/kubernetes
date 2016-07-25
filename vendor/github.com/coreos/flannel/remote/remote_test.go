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

package remote

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"sync"
	"syscall"
	"testing"
	"time"

	"golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

const expectedNetwork = "10.1.0.0/16"

type fixture struct {
	ctx      context.Context
	cancel   context.CancelFunc
	srvAddr  string
	registry *subnet.MockSubnetRegistry
	sm       subnet.Manager
	wg       sync.WaitGroup
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}

	config := fmt.Sprintf(`{"Network": %q}`, expectedNetwork)
	f.registry = subnet.NewMockRegistry("", config, nil)
	sm := subnet.NewMockManager(f.registry)

	f.srvAddr = "127.0.0.1:9999"

	f.ctx, f.cancel = context.WithCancel(context.Background())
	f.wg.Add(1)
	go func() {
		RunServer(f.ctx, sm, f.srvAddr, "", "", "")
		f.wg.Done()
	}()

	var err error
	f.sm, err = NewRemoteManager(f.srvAddr, "", "", "")
	if err != nil {
		panic(fmt.Sprintf("Failed to create remote mananager: %v", err))
	}

	for i := 0; ; i++ {
		_, err := f.sm.GetNetworkConfig(f.ctx, "_")
		if err == nil {
			break
		}

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

	return f
}

func (f *fixture) Close() {
	f.cancel()
	f.wg.Wait()
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
			if oserr, ok := operr.Err.(*os.SyscallError); ok {
				return oserr.Err == syscall.ECONNREFUSED
			}
			return operr.Err == syscall.ECONNREFUSED
		}
	}
	return false
}

func TestGetConfig(t *testing.T) {
	f := newFixture(t)
	defer f.Close()

	cfg, err := f.sm.GetNetworkConfig(f.ctx, "_")
	if err != nil {
		t.Fatalf("GetNetworkConfig failed: %v", err)
	}

	if cfg.Network.String() != expectedNetwork {
		t.Errorf("GetNetworkConfig returned bad network: %v vs %v", cfg.Network, expectedNetwork)
	}
}

func TestAcquireRenewLease(t *testing.T) {
	f := newFixture(t)
	defer f.Close()

	attrs := &subnet.LeaseAttrs{
		PublicIP: mustParseIP4("1.1.1.1"),
	}

	l, err := f.sm.AcquireLease(f.ctx, "_", attrs)
	if err != nil {
		t.Fatalf("AcquireLease failed: %v", err)
	}

	if !mustParseIP4Net(expectedNetwork).Contains(l.Subnet.IP) {
		t.Errorf("AcquireLease returned subnet not in network: %v (in %v)", l.Subnet, expectedNetwork)
	}

	if err = f.sm.RenewLease(f.ctx, "_", l); err != nil {
		t.Errorf("RenewLease failed: %v", err)
	}
}

func TestWatchLeases(t *testing.T) {
	f := newFixture(t)
	defer f.Close()

	events := make(chan []subnet.Event)
	f.wg.Add(1)
	go func() {
		subnet.WatchLeases(f.ctx, f.sm, "_", nil, events)
		f.wg.Done()
	}()

	attrs := &subnet.LeaseAttrs{
		PublicIP: mustParseIP4("1.1.1.2"),
	}
	l, err := f.sm.AcquireLease(f.ctx, "_", attrs)
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

func TestRevokeLease(t *testing.T) {
	f := newFixture(t)
	defer f.Close()

	attrs := &subnet.LeaseAttrs{
		PublicIP: mustParseIP4("1.1.1.1"),
	}

	l, err := f.sm.AcquireLease(f.ctx, "_", attrs)
	if err != nil {
		t.Fatalf("AcquireLease failed: %v", err)
	}

	if err := f.sm.RevokeLease(f.ctx, "_", l.Subnet); err != nil {
		t.Fatalf("RevokeLease failed: %v", err)
	}

	_, err = f.sm.WatchLease(f.ctx, "_", l.Subnet, nil)
	if err == nil {
		t.Fatalf("Revoked lease found")
	}
}

func TestWatchNetworks(t *testing.T) {
	f := newFixture(t)
	defer f.Close()

	events := make(chan []subnet.Event)
	f.wg.Add(1)
	go func() {
		subnet.WatchNetworks(f.ctx, f.sm, events)
		f.wg.Done()
	}()

	// skip over the initial snapshot
	<-events

	expectedNetname := "foobar"
	config := fmt.Sprintf(`{"Network": %q}`, expectedNetwork)
	err := f.registry.CreateNetwork(f.ctx, expectedNetname, config)
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

	err = f.registry.DeleteNetwork(f.ctx, expectedNetname)
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
