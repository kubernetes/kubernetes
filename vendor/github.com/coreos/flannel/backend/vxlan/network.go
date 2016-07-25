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

package vxlan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	log "github.com/golang/glog"
	"github.com/vishvananda/netlink"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

type network struct {
	backend.SimpleNetwork
	name     string
	extIface *backend.ExternalInterface
	dev      *vxlanDevice
	rts      routes
	sm       subnet.Manager
}

func newNetwork(name string, sm subnet.Manager, extIface *backend.ExternalInterface, dev *vxlanDevice, nw ip.IP4Net, l *subnet.Lease) (*network, error) {
	n := &network{
		SimpleNetwork: backend.SimpleNetwork{
			SubnetLease: l,
			ExtIface:    extIface,
		},
		name: name,
		sm:   sm,
		dev:  dev,
	}

	return n, nil
}

func (n *network) Run(ctx context.Context) {
	log.Info("Watching for L3 misses")
	misses := make(chan *netlink.Neigh, 100)
	// Unfrtunately MonitorMisses does not take a cancel channel
	// as there's no wait to interrupt netlink socket recv
	go n.dev.MonitorMisses(misses)

	wg := sync.WaitGroup{}

	log.Info("Watching for new subnet leases")
	evts := make(chan []subnet.Event)
	wg.Add(1)
	go func() {
		subnet.WatchLeases(ctx, n.sm, n.name, n.SubnetLease, evts)
		log.Info("WatchLeases exited")
		wg.Done()
	}()

	defer wg.Wait()
	initialEvtsBatch := <-evts
	for {
		err := n.handleInitialSubnetEvents(initialEvtsBatch)
		if err == nil {
			break
		}
		log.Error(err, " About to retry")
		time.Sleep(time.Second)
	}

	for {
		select {
		case miss := <-misses:
			n.handleMiss(miss)

		case evtBatch := <-evts:
			n.handleSubnetEvents(evtBatch)

		case <-ctx.Done():
			return
		}
	}
}

func (n *network) MTU() int {
	return n.dev.MTU()
}

type vxlanLeaseAttrs struct {
	VtepMAC hardwareAddr
}

func (n *network) handleSubnetEvents(batch []subnet.Event) {
	for _, evt := range batch {
		switch evt.Type {
		case subnet.EventAdded:
			log.Info("Subnet added: ", evt.Lease.Subnet)

			if evt.Lease.Attrs.BackendType != "vxlan" {
				log.Warningf("Ignoring non-vxlan subnet: type=%v", evt.Lease.Attrs.BackendType)
				continue
			}

			var attrs vxlanLeaseAttrs
			if err := json.Unmarshal(evt.Lease.Attrs.BackendData, &attrs); err != nil {
				log.Error("Error decoding subnet lease JSON: ", err)
				continue
			}
			n.rts.set(evt.Lease.Subnet, net.HardwareAddr(attrs.VtepMAC))
			n.dev.AddL2(neigh{IP: evt.Lease.Attrs.PublicIP, MAC: net.HardwareAddr(attrs.VtepMAC)})

		case subnet.EventRemoved:
			log.Info("Subnet removed: ", evt.Lease.Subnet)

			if evt.Lease.Attrs.BackendType != "vxlan" {
				log.Warningf("Ignoring non-vxlan subnet: type=%v", evt.Lease.Attrs.BackendType)
				continue
			}

			var attrs vxlanLeaseAttrs
			if err := json.Unmarshal(evt.Lease.Attrs.BackendData, &attrs); err != nil {
				log.Error("Error decoding subnet lease JSON: ", err)
				continue
			}

			if len(attrs.VtepMAC) > 0 {
				n.dev.DelL2(neigh{IP: evt.Lease.Attrs.PublicIP, MAC: net.HardwareAddr(attrs.VtepMAC)})
			}
			n.rts.remove(evt.Lease.Subnet)

		default:
			log.Error("Internal error: unknown event type: ", int(evt.Type))
		}
	}
}

func (n *network) handleInitialSubnetEvents(batch []subnet.Event) error {
	log.Infof("Handling initial subnet events")
	fdbTable, err := n.dev.GetL2List()
	if err != nil {
		return fmt.Errorf("error fetching L2 table: %v", err)
	}

	for _, fdbEntry := range fdbTable {
		log.Infof("fdb already populated with: %s %s ", fdbEntry.IP, fdbEntry.HardwareAddr)
	}

	evtMarker := make([]bool, len(batch))
	leaseAttrsList := make([]vxlanLeaseAttrs, len(batch))
	fdbEntryMarker := make([]bool, len(fdbTable))

	for i, evt := range batch {
		if evt.Lease.Attrs.BackendType != "vxlan" {
			log.Warningf("Ignoring non-vxlan subnet: type=%v", evt.Lease.Attrs.BackendType)
			evtMarker[i] = true
			continue
		}

		if err := json.Unmarshal(evt.Lease.Attrs.BackendData, &leaseAttrsList[i]); err != nil {
			log.Error("Error decoding subnet lease JSON: ", err)
			evtMarker[i] = true
			continue
		}

		for j, fdbEntry := range fdbTable {
			if evt.Lease.Attrs.PublicIP.ToIP().Equal(fdbEntry.IP) && bytes.Equal([]byte(leaseAttrsList[i].VtepMAC), []byte(fdbEntry.HardwareAddr)) {
				evtMarker[i] = true
				fdbEntryMarker[j] = true
				break
			}
		}
		n.rts.set(evt.Lease.Subnet, net.HardwareAddr(leaseAttrsList[i].VtepMAC))
	}

	for j, marker := range fdbEntryMarker {
		if !marker && fdbTable[j].IP != nil {
			err := n.dev.DelL2(neigh{IP: ip.FromIP(fdbTable[j].IP), MAC: fdbTable[j].HardwareAddr})
			if err != nil {
				log.Error("Delete L2 failed: ", err)
			}
		}
	}

	for i, marker := range evtMarker {
		if !marker {
			err := n.dev.AddL2(neigh{IP: batch[i].Lease.Attrs.PublicIP, MAC: net.HardwareAddr(leaseAttrsList[i].VtepMAC)})
			if err != nil {
				log.Error("Add L2 failed: ", err)
			}

		}
	}
	return nil
}

func (n *network) handleMiss(miss *netlink.Neigh) {
	switch {
	case len(miss.IP) == 0 && len(miss.HardwareAddr) == 0:
		log.Info("Ignoring nil miss")

	case len(miss.HardwareAddr) == 0:
		n.handleL3Miss(miss)

	default:
		log.Infof("Ignoring not a miss: %v, %v", miss.HardwareAddr, miss.IP)
	}
}

func (n *network) handleL3Miss(miss *netlink.Neigh) {
	log.Infof("L3 miss: %v", miss.IP)

	rt := n.rts.findByNetwork(ip.FromIP(miss.IP))
	if rt == nil {
		log.Infof("Route for %v not found", miss.IP)
		return
	}

	if err := n.dev.AddL3(neigh{IP: ip.FromIP(miss.IP), MAC: rt.vtepMAC}); err != nil {
		log.Errorf("AddL3 failed: %v", err)
	} else {
		log.Info("AddL3 succeeded")
	}
}
