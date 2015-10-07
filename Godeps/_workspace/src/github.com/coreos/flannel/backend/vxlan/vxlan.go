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

package vxlan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/vishvananda/netlink"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

const (
	defaultVNI = 1
)

type VXLANBackend struct {
	sm      subnet.Manager
	network string
	cfg     struct {
		VNI  int
		Port int
	}
	extIndex int
	extIaddr net.IP
	extEaddr net.IP
	lease    *subnet.Lease
	dev      *vxlanDevice
	rts      routes
}

func New(sm subnet.Manager, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error) {
	vb := &VXLANBackend{
		sm:       sm,
		extIndex: extIface.Index,
		extIaddr: extIaddr,
		extEaddr: extEaddr,
	}
	vb.cfg.VNI = defaultVNI

	return vb, nil
}

func newSubnetAttrs(extEaddr net.IP, mac net.HardwareAddr) (*subnet.LeaseAttrs, error) {
	data, err := json.Marshal(&vxlanLeaseAttrs{hardwareAddr(mac)})
	if err != nil {
		return nil, err
	}

	return &subnet.LeaseAttrs{
		PublicIP:    ip.FromIP(extEaddr),
		BackendType: "vxlan",
		BackendData: json.RawMessage(data),
	}, nil
}

func (vb *VXLANBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (*backend.SubnetDef, error) {
	vb.network = network

	// Parse our configuration
	if len(config.Backend) > 0 {
		if err := json.Unmarshal(config.Backend, &vb.cfg); err != nil {
			return nil, fmt.Errorf("error decoding VXLAN backend config: %v", err)
		}
	}

	devAttrs := vxlanDeviceAttrs{
		vni:       uint32(vb.cfg.VNI),
		name:      fmt.Sprintf("flannel.%v", vb.cfg.VNI),
		vtepIndex: vb.extIndex,
		vtepAddr:  vb.extIaddr,
		vtepPort:  vb.cfg.Port,
	}

	var err error
	for {
		vb.dev, err = newVXLANDevice(&devAttrs)
		if err == nil {
			break
		} else {
			log.Error("VXLAN init: ", err)
			log.Info("Retrying in 1 second...")

			// wait 1 sec before retrying
			time.Sleep(1 * time.Second)
		}
	}

	sa, err := newSubnetAttrs(vb.extEaddr, vb.dev.MACAddr())
	if err != nil {
		return nil, err
	}

	l, err := vb.sm.AcquireLease(ctx, vb.network, sa)
	switch err {
	case nil:
		vb.lease = l

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}

	// vxlan's subnet is that of the whole overlay network (e.g. /16)
	// and not that of the individual host (e.g. /24)
	vxlanNet := ip.IP4Net{
		IP:        l.Subnet.IP,
		PrefixLen: config.Network.PrefixLen,
	}
	if err = vb.dev.Configure(vxlanNet); err != nil {
		return nil, err
	}

	return &backend.SubnetDef{
		Lease: l,
		MTU:   vb.dev.MTU(),
	}, nil
}

func (vb *VXLANBackend) Run(ctx context.Context) {
	log.Info("Watching for L3 misses")
	misses := make(chan *netlink.Neigh, 100)
	// Unfrtunately MonitorMisses does not take a cancel channel
	// as there's no wait to interrupt netlink socket recv
	go vb.dev.MonitorMisses(misses)

	wg := sync.WaitGroup{}

	log.Info("Watching for new subnet leases")
	evts := make(chan []subnet.Event)
	wg.Add(1)
	go func() {
		subnet.WatchLeases(ctx, vb.sm, vb.network, vb.lease, evts)
		log.Info("WatchLeases exited")
		wg.Done()
	}()

	defer wg.Wait()
	initialEvtsBatch := <-evts
	for {
		err := vb.handleInitialSubnetEvents(initialEvtsBatch)
		if err == nil {
			break
		}
		log.Error(err, " About to retry")
		time.Sleep(time.Second)
	}

	for {
		select {
		case miss := <-misses:
			vb.handleMiss(miss)

		case evtBatch := <-evts:
			vb.handleSubnetEvents(evtBatch)

		case <-ctx.Done():
			return
		}
	}
}

// So we can make it JSON (un)marshalable
type hardwareAddr net.HardwareAddr

func (hw hardwareAddr) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf("%q", net.HardwareAddr(hw))), nil
}

func (hw *hardwareAddr) UnmarshalJSON(b []byte) error {
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		return fmt.Errorf("error parsing hardware addr")
	}

	b = b[1 : len(b)-1]

	mac, err := net.ParseMAC(string(b))
	if err != nil {
		return err
	}

	*hw = hardwareAddr(mac)
	return nil
}

type vxlanLeaseAttrs struct {
	VtepMAC hardwareAddr
}

func (vb *VXLANBackend) handleSubnetEvents(batch []subnet.Event) {
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
			vb.rts.set(evt.Lease.Subnet, net.HardwareAddr(attrs.VtepMAC))
			vb.dev.AddL2(neigh{IP: evt.Lease.Attrs.PublicIP, MAC: net.HardwareAddr(attrs.VtepMAC)})

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
				vb.dev.DelL2(neigh{IP: evt.Lease.Attrs.PublicIP, MAC: net.HardwareAddr(attrs.VtepMAC)})
			}
			vb.rts.remove(evt.Lease.Subnet)

		default:
			log.Error("Internal error: unknown event type: ", int(evt.Type))
		}
	}
}

func (vb *VXLANBackend) handleInitialSubnetEvents(batch []subnet.Event) error {
	log.Infof("Handling initial subnet events")
	fdbTable, err := vb.dev.GetL2List()
	if err != nil {
		return fmt.Errorf("Error fetching L2 table: %v", err)
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
		vb.rts.set(evt.Lease.Subnet, net.HardwareAddr(leaseAttrsList[i].VtepMAC))
	}

	for j, marker := range fdbEntryMarker {
		if !marker && fdbTable[j].IP != nil {
			err := vb.dev.DelL2(neigh{IP: ip.FromIP(fdbTable[j].IP), MAC: fdbTable[j].HardwareAddr})
			if err != nil {
				log.Error("Delete L2 failed: ", err)
			}
		}
	}

	for i, marker := range evtMarker {
		if !marker {
			err := vb.dev.AddL2(neigh{IP: batch[i].Lease.Attrs.PublicIP, MAC: net.HardwareAddr(leaseAttrsList[i].VtepMAC)})
			if err != nil {
				log.Error("Add L2 failed: ", err)
			}

		}
	}
	return nil
}

func (vb *VXLANBackend) handleMiss(miss *netlink.Neigh) {
	switch {
	case len(miss.IP) == 0 && len(miss.HardwareAddr) == 0:
		log.Info("Ignoring nil miss")

	case len(miss.HardwareAddr) == 0:
		vb.handleL3Miss(miss)

	default:
		log.Infof("Ignoring not a miss: %v, %v", miss.HardwareAddr, miss.IP)
	}
}

func (vb *VXLANBackend) handleL3Miss(miss *netlink.Neigh) {
	log.Infof("L3 miss: %v", miss.IP)

	rt := vb.rts.findByNetwork(ip.FromIP(miss.IP))
	if rt == nil {
		log.Infof("Route for %v not found", miss.IP)
		return
	}

	if err := vb.dev.AddL3(neigh{IP: ip.FromIP(miss.IP), MAC: rt.vtepMAC}); err != nil {
		log.Errorf("AddL3 failed: %v", err)
	} else {
		log.Info("AddL3 succeeded")
	}
}
