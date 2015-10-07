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

package udp

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"sync"
	"syscall"

	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/vishvananda/netlink"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

const (
	encapOverhead = 28 // 20 bytes IP hdr + 8 bytes UDP hdr
	defaultPort   = 8285
)

type UdpBackend struct {
	sm       subnet.Manager
	network  string
	publicIP ip.IP4
	cfg      struct {
		 Port int
	}
	lease    *subnet.Lease
	ctl      *os.File
	ctl2     *os.File
	tun      *os.File
	conn     *net.UDPConn
	mtu      int
	tunNet   ip.IP4Net
}

func New(sm subnet.Manager, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error) {
	be := UdpBackend{
		sm:       sm,
		publicIP: ip.FromIP(extEaddr),
		// TUN MTU will be smaller b/c of encap (IP+UDP hdrs)
		mtu:      extIface.MTU - encapOverhead,
	}
	be.cfg.Port = defaultPort
	return &be, nil
}

func (m *UdpBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (*backend.SubnetDef, error) {
	m.network = network

	// Parse our configuration
	if len(config.Backend) > 0 {
		if err := json.Unmarshal(config.Backend, &m.cfg); err != nil {
			return nil, fmt.Errorf("error decoding UDP backend config: %v", err)
		}
	}

	// Acquire the lease form subnet manager
	attrs := subnet.LeaseAttrs{
		PublicIP: m.publicIP,
	}

	l, err := m.sm.AcquireLease(ctx, m.network, &attrs)
	switch err {
	case nil:
		m.lease = l

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}

	// Tunnel's subnet is that of the whole overlay network (e.g. /16)
	// and not that of the individual host (e.g. /24)
	m.tunNet = ip.IP4Net{
		IP:        l.Subnet.IP,
		PrefixLen: config.Network.PrefixLen,
	}

	if err = m.initTun(); err != nil {
		return nil, err
	}

	m.conn, err = net.ListenUDP("udp4", &net.UDPAddr{Port: m.cfg.Port})
	if err != nil {
		return nil, fmt.Errorf("failed to start listening on UDP socket: %v", err)
	}

	m.ctl, m.ctl2, err = newCtlSockets()
	if err != nil {
		return nil, fmt.Errorf("failed to create control socket: %v", err)
	}

	return &backend.SubnetDef{
		Lease: l,
		MTU:   m.mtu,
	}, nil
}

func (m *UdpBackend) Run(ctx context.Context) {
	// one for each goroutine below
	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		runCProxy(m.tun, m.conn, m.ctl2, m.tunNet.IP, m.mtu)
		wg.Done()
	}()

	log.Info("Watching for new subnet leases")

	evts := make(chan []subnet.Event)

	wg.Add(1)
	go func() {
		subnet.WatchLeases(ctx, m.sm, m.network, m.lease, evts)
		wg.Done()
	}()

	for {
		select {
		case evtBatch := <-evts:
			m.processSubnetEvents(evtBatch)

		case <-ctx.Done():
			stopProxy(m.ctl)
			break
		}
	}

	wg.Wait()
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

func (m *UdpBackend) initTun() error {
	var tunName string
	var err error

	m.tun, tunName, err = ip.OpenTun("flannel%d")
	if err != nil {
		return fmt.Errorf("Failed to open TUN device: %v", err)
	}

	err = configureIface(tunName, m.tunNet, m.mtu)
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

	err = netlink.AddrAdd(iface, &netlink.Addr{ipn.ToIPNet(), ""})
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
		return fmt.Errorf("Failed to add route (%v -> %v): %v", ipn.Network().String(), ifname, err)
	}

	return nil
}

func (m *UdpBackend) processSubnetEvents(batch []subnet.Event) {
	for _, evt := range batch {
		switch evt.Type {
		case subnet.EventAdded:
			log.Info("Subnet added: ", evt.Lease.Subnet)

			setRoute(m.ctl, evt.Lease.Subnet, evt.Lease.Attrs.PublicIP, m.cfg.Port)

		case subnet.EventRemoved:
			log.Info("Subnet removed: ", evt.Lease.Subnet)

			removeRoute(m.ctl, evt.Lease.Subnet)

		default:
			log.Error("Internal error: unknown event type: ", int(evt.Type))
		}
	}
}
