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

package udp

import (
	"fmt"
	"net"
	"os"
	"sync"
	"syscall"

	log "github.com/golang/glog"
	"github.com/vishvananda/netlink"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

const (
	encapOverhead = 28 // 20 bytes IP hdr + 8 bytes UDP hdr
)

type network struct {
	backend.SimpleNetwork
	name   string
	port   int
	ctl    *os.File
	ctl2   *os.File
	tun    *os.File
	conn   *net.UDPConn
	tunNet ip.IP4Net
	sm     subnet.Manager
}

func newNetwork(name string, sm subnet.Manager, extIface *backend.ExternalInterface, port int, nw ip.IP4Net, l *subnet.Lease) (*network, error) {
	n := &network{
		SimpleNetwork: backend.SimpleNetwork{
			SubnetLease: l,
			ExtIface:    extIface,
		},
		name: name,
		port: port,
		sm:   sm,
	}

	n.tunNet = nw

	if err := n.initTun(); err != nil {
		return nil, err
	}

	var err error
	n.conn, err = net.ListenUDP("udp4", &net.UDPAddr{IP: extIface.IfaceAddr, Port: port})
	if err != nil {
		return nil, fmt.Errorf("failed to start listening on UDP socket: %v", err)
	}

	n.ctl, n.ctl2, err = newCtlSockets()
	if err != nil {
		return nil, fmt.Errorf("failed to create control socket: %v", err)
	}

	return n, nil
}

func (n *network) Run(ctx context.Context) {
	defer func() {
		n.tun.Close()
		n.conn.Close()
		n.ctl.Close()
		n.ctl2.Close()
	}()

	// one for each goroutine below
	wg := sync.WaitGroup{}
	defer wg.Wait()

	wg.Add(1)
	go func() {
		runCProxy(n.tun, n.conn, n.ctl2, n.tunNet.IP, n.MTU())
		wg.Done()
	}()

	log.Info("Watching for new subnet leases")

	evts := make(chan []subnet.Event)

	wg.Add(1)
	go func() {
		subnet.WatchLeases(ctx, n.sm, n.name, n.SubnetLease, evts)
		wg.Done()
	}()

	for {
		select {
		case evtBatch := <-evts:
			n.processSubnetEvents(evtBatch)

		case <-ctx.Done():
			stopProxy(n.ctl)
			return
		}
	}
}

func (n *network) MTU() int {
	return n.ExtIface.Iface.MTU - encapOverhead
}

func newCtlSockets() (*os.File, *os.File, error) {
	fds, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_SEQPACKET, 0)
	if err != nil {
		return nil, nil, err
	}

	f1 := os.NewFile(uintptr(fds[0]), "ctl")
	f2 := os.NewFile(uintptr(fds[1]), "ctl")
	return f1, f2, nil
}

func (n *network) initTun() error {
	var tunName string
	var err error

	n.tun, tunName, err = ip.OpenTun("flannel%d")
	if err != nil {
		return fmt.Errorf("failed to open TUN device: %v", err)
	}

	err = configureIface(tunName, n.tunNet, n.MTU())
	if err != nil {
		return err
	}

	return nil
}

func configureIface(ifname string, ipn ip.IP4Net, mtu int) error {
	iface, err := netlink.LinkByName(ifname)
	if err != nil {
		return fmt.Errorf("failed to lookup interface %v", ifname)
	}

	err = netlink.AddrAdd(iface, &netlink.Addr{IPNet: ipn.ToIPNet(), Label: ""})
	if err != nil {
		return fmt.Errorf("failed to add IP address %v to %v: %v", ipn.String(), ifname, err)
	}

	err = netlink.LinkSetMTU(iface, mtu)
	if err != nil {
		return fmt.Errorf("failed to set MTU for %v: %v", ifname, err)
	}

	err = netlink.LinkSetUp(iface)
	if err != nil {
		return fmt.Errorf("failed to set interface %v to UP state: %v", ifname, err)
	}

	// explicitly add a route since there might be a route for a subnet already
	// installed by Docker and then it won't get auto added
	err = netlink.RouteAdd(&netlink.Route{
		LinkIndex: iface.Attrs().Index,
		Scope:     netlink.SCOPE_UNIVERSE,
		Dst:       ipn.Network().ToIPNet(),
	})
	if err != nil && err != syscall.EEXIST {
		return fmt.Errorf("failed to add route (%v -> %v): %v", ipn.Network().String(), ifname, err)
	}

	return nil
}

func (n *network) processSubnetEvents(batch []subnet.Event) {
	for _, evt := range batch {
		switch evt.Type {
		case subnet.EventAdded:
			log.Info("Subnet added: ", evt.Lease.Subnet)

			setRoute(n.ctl, evt.Lease.Subnet, evt.Lease.Attrs.PublicIP, n.port)

		case subnet.EventRemoved:
			log.Info("Subnet removed: ", evt.Lease.Subnet)

			removeRoute(n.ctl, evt.Lease.Subnet)

		default:
			log.Error("Internal error: unknown event type: ", int(evt.Type))
		}
	}
}
