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
	"time"

	log "github.com/golang/glog"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
)

// WatchLeases performs a long term watch of the given network's subnet leases
// and communicates addition/deletion events on receiver channel. It takes care
// of handling "fall-behind" logic where the history window has advanced too far
// and it needs to diff the latest snapshot with its saved state and generate events
func WatchLeases(ctx context.Context, sm Manager, network string, ownLease *Lease, receiver chan []Event) {
	lw := &leaseWatcher{
		ownLease: ownLease,
	}
	var cursor interface{}

	for {
		res, err := sm.WatchLeases(ctx, network, cursor)
		if err != nil {
			if err == context.Canceled || err == context.DeadlineExceeded {
				return
			}

			log.Errorf("Watch subnets: %v", err)
			time.Sleep(time.Second)
			continue
		}

		cursor = res.Cursor

		batch := []Event{}

		if len(res.Events) > 0 {
			batch = lw.update(res.Events)
		} else {
			batch = lw.reset(res.Snapshot)
		}

		if len(batch) > 0 {
			receiver <- batch
		}
	}
}

type leaseWatcher struct {
	ownLease *Lease
	leases   []Lease
}

func (lw *leaseWatcher) reset(leases []Lease) []Event {
	batch := []Event{}

	for _, nl := range leases {
		if lw.ownLease != nil && nl.Subnet.Equal(lw.ownLease.Subnet) {
			continue
		}

		found := false
		for i, ol := range lw.leases {
			if ol.Subnet.Equal(nl.Subnet) {
				lw.leases = deleteLease(lw.leases, i)
				found = true
				break
			}
		}

		if !found {
			// new lease
			batch = append(batch, Event{EventAdded, nl, ""})
		}
	}

	// everything left in sm.leases has been deleted
	for _, l := range lw.leases {
		if lw.ownLease != nil && l.Subnet.Equal(lw.ownLease.Subnet) {
			continue
		}
		batch = append(batch, Event{EventRemoved, l, ""})
	}

	// copy the leases over (caution: don't just assign a slice)
	lw.leases = make([]Lease, len(leases))
	copy(lw.leases, leases)

	return batch
}

func (lw *leaseWatcher) update(events []Event) []Event {
	batch := []Event{}

	for _, e := range events {
		if lw.ownLease != nil && e.Lease.Subnet.Equal(lw.ownLease.Subnet) {
			continue
		}

		switch e.Type {
		case EventAdded:
			batch = append(batch, lw.add(&e.Lease))

		case EventRemoved:
			batch = append(batch, lw.remove(&e.Lease))
		}
	}

	return batch
}

func (lw *leaseWatcher) add(lease *Lease) Event {
	for i, l := range lw.leases {
		if l.Subnet.Equal(lease.Subnet) {
			lw.leases[i] = *lease
			return Event{EventAdded, lw.leases[i], ""}
		}
	}

	lw.leases = append(lw.leases, *lease)

	return Event{EventAdded, lw.leases[len(lw.leases)-1], ""}
}

func (lw *leaseWatcher) remove(lease *Lease) Event {
	for i, l := range lw.leases {
		if l.Subnet.Equal(lease.Subnet) {
			lw.leases = deleteLease(lw.leases, i)
			return Event{EventRemoved, l, ""}
		}
	}

	log.Errorf("Removed subnet (%s) was not found", lease.Subnet)
	return Event{EventRemoved, *lease, ""}
}

func deleteLease(l []Lease, i int) []Lease {
	l[i] = l[len(l)-1]
	return l[:len(l)-1]
}

// WatchNetworks performs a long term watch of flannel networks and communicates
// addition/deletion events on receiver channel. It takes care of handling
// "fall-behind" logic where the history window has advanced too far and it
// needs to diff the latest snapshot with its saved state and generate events
func WatchNetworks(ctx context.Context, sm Manager, receiver chan []Event) {
	nw := newNetWatcher()
	var cursor interface{}

	for {
		res, err := sm.WatchNetworks(ctx, cursor)
		if err != nil {
			if err == context.Canceled || err == context.DeadlineExceeded {
				return
			}

			log.Errorf("Watch networks: %v", err)
			time.Sleep(time.Second)
			continue
		}
		cursor = res.Cursor

		batch := []Event{}

		if len(res.Events) > 0 {
			batch = nw.update(res.Events)
		} else {
			batch = nw.reset(res.Snapshot)
		}

		if len(batch) > 0 {
			receiver <- batch
		}
	}
}

type netWatcher struct {
	networks map[string]bool
}

func newNetWatcher() *netWatcher {
	return &netWatcher{networks: make(map[string]bool)}
}

func (nw *netWatcher) reset(networks []string) []Event {
	batch := []Event{}
	newNetworks := make(map[string]bool)

	for _, netname := range networks {
		if nw.networks[netname] {
			delete(nw.networks, netname)
		} else {
			// new network
			batch = append(batch, Event{EventAdded, Lease{}, netname})
		}
		newNetworks[netname] = true
	}

	// everything left in sm.networks has been deleted
	for netname, _ := range nw.networks {
		batch = append(batch, Event{EventRemoved, Lease{}, netname})
	}

	nw.networks = newNetworks

	return batch
}

func (nw *netWatcher) update(events []Event) []Event {
	batch := []Event{}

	for _, e := range events {
		switch e.Type {
		case EventAdded:
			batch = append(batch, nw.add(e.Network))

		case EventRemoved:
			batch = append(batch, nw.remove(e.Network))
		}
	}

	return batch
}

func (nw *netWatcher) add(network string) Event {
	if _, ok := nw.networks[network]; !ok {
		nw.networks[network] = true
	}

	return Event{EventAdded, Lease{}, network}
}

func (nw *netWatcher) remove(network string) Event {
	if _, ok := nw.networks[network]; ok {
		delete(nw.networks, network)
	} else {
		log.Errorf("Removed network (%s) was not found", network)
	}

	return Event{EventRemoved, Lease{}, network}
}

// WatchLease performs a long term watch of the given network's subnet lease
// and communicates addition/deletion events on receiver channel. It takes care
// of handling "fall-behind" logic where the history window has advanced too far
// and it needs to diff the latest snapshot with its saved state and generate events
func WatchLease(ctx context.Context, sm Manager, network string, sn ip.IP4Net, receiver chan Event) {
	var cursor interface{}

	for {
		wr, err := sm.WatchLease(ctx, network, sn, cursor)
		if err != nil {
			if err == context.Canceled || err == context.DeadlineExceeded {
				return
			}

			log.Errorf("Subnet watch failed: %v", err)
			time.Sleep(time.Second)
			continue
		}

		if len(wr.Snapshot) > 0 {
			receiver <- Event{
				Type:  EventAdded,
				Lease: wr.Snapshot[0],
			}
		} else {
			receiver <- wr.Events[0]
		}

		cursor = wr.Cursor
	}
}
