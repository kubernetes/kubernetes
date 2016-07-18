// Copyright 2015 flannel authors
// Copyright 2015 Red Hat, Inc.
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

package network

import (
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/coreos/go-systemd/daemon"
	log "github.com/golang/glog"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

type CmdLineOpts struct {
	publicIP      string
	ipMasq        bool
	subnetFile    string
	subnetDir     string
	iface         string
	networks      string
	watchNetworks bool
}

var errAlreadyExists = errors.New("already exists")

var opts CmdLineOpts

func init() {
	flag.StringVar(&opts.publicIP, "public-ip", "", "IP accessible by other nodes for inter-host communication")
	flag.StringVar(&opts.subnetFile, "subnet-file", "/run/flannel/subnet.env", "filename where env variables (subnet, MTU, ... ) will be written to")
	flag.StringVar(&opts.subnetDir, "subnet-dir", "/run/flannel/networks", "directory where files with env variables (subnet, MTU, ...) will be written to")
	flag.StringVar(&opts.iface, "iface", "", "interface to use (IP or name) for inter-host communication")
	flag.StringVar(&opts.networks, "networks", "", "run in multi-network mode and service the specified networks")
	flag.BoolVar(&opts.watchNetworks, "watch-networks", false, "run in multi-network mode and watch for networks from 'networks' or all networks")
	flag.BoolVar(&opts.ipMasq, "ip-masq", false, "setup IP masquerade rule for traffic destined outside of overlay network")
}

type Manager struct {
	ctx             context.Context
	sm              subnet.Manager
	bm              backend.Manager
	allowedNetworks map[string]bool
	mux             sync.Mutex
	networks        map[string]*Network
	watch           bool
	ipMasq          bool
	extIface        *backend.ExternalInterface
}

func (m *Manager) isNetAllowed(name string) bool {
	// If allowedNetworks is empty all networks are allowed
	if len(m.allowedNetworks) > 0 {
		_, ok := m.allowedNetworks[name]
		return ok
	}
	return true
}

func (m *Manager) isMultiNetwork() bool {
	return len(m.allowedNetworks) > 0 || m.watch
}

func NewNetworkManager(ctx context.Context, sm subnet.Manager) (*Manager, error) {
	extIface, err := lookupExtIface(opts.iface)
	if err != nil {
		return nil, err
	}

	bm := backend.NewManager(ctx, sm, extIface)

	manager := &Manager{
		ctx:             ctx,
		sm:              sm,
		bm:              bm,
		allowedNetworks: make(map[string]bool),
		networks:        make(map[string]*Network),
		watch:           opts.watchNetworks,
		ipMasq:          opts.ipMasq,
		extIface:        extIface,
	}

	for _, name := range strings.Split(opts.networks, ",") {
		if name != "" {
			manager.allowedNetworks[name] = true
		}
	}

	return manager, nil
}

func lookupExtIface(ifname string) (*backend.ExternalInterface, error) {
	var iface *net.Interface
	var iaddr net.IP
	var err error

	if len(ifname) > 0 {
		if iaddr = net.ParseIP(ifname); iaddr != nil {
			iface, err = ip.GetInterfaceByIP(iaddr)
			if err != nil {
				return nil, fmt.Errorf("error looking up interface %s: %s", ifname, err)
			}
		} else {
			iface, err = net.InterfaceByName(ifname)
			if err != nil {
				return nil, fmt.Errorf("error looking up interface %s: %s", ifname, err)
			}
		}
	} else {
		log.Info("Determining IP address of default interface")
		if iface, err = ip.GetDefaultGatewayIface(); err != nil {
			return nil, fmt.Errorf("failed to get default interface: %s", err)
		}
	}

	if iaddr == nil {
		iaddr, err = ip.GetIfaceIP4Addr(iface)
		if err != nil {
			return nil, fmt.Errorf("failed to find IPv4 address for interface %s", iface.Name)
		}
	}

	if iface.MTU == 0 {
		return nil, fmt.Errorf("failed to determine MTU for %s interface", iaddr)
	}

	var eaddr net.IP

	if len(opts.publicIP) > 0 {
		eaddr = net.ParseIP(opts.publicIP)
		if eaddr == nil {
			return nil, fmt.Errorf("invalid public IP address", opts.publicIP)
		}
	}

	if eaddr == nil {
		eaddr = iaddr
	}

	log.Infof("Using %s as external interface", iaddr)
	log.Infof("Using %s as external endpoint", eaddr)

	return &backend.ExternalInterface{
		Iface:     iface,
		IfaceAddr: iaddr,
		ExtAddr:   eaddr,
	}, nil
}

func writeSubnetFile(path string, nw ip.IP4Net, ipMasq bool, bn backend.Network) error {
	dir, name := filepath.Split(path)
	os.MkdirAll(dir, 0755)

	tempFile := filepath.Join(dir, "."+name)
	f, err := os.Create(tempFile)
	if err != nil {
		return err
	}

	// Write out the first usable IP by incrementing
	// sn.IP by one
	sn := bn.Lease().Subnet
	sn.IP += 1

	fmt.Fprintf(f, "FLANNEL_NETWORK=%s\n", nw)
	fmt.Fprintf(f, "FLANNEL_SUBNET=%s\n", sn)
	fmt.Fprintf(f, "FLANNEL_MTU=%d\n", bn.MTU())
	_, err = fmt.Fprintf(f, "FLANNEL_IPMASQ=%v\n", ipMasq)
	f.Close()
	if err != nil {
		return err
	}

	// rename(2) the temporary file to the desired location so that it becomes
	// atomically visible with the contents
	return os.Rename(tempFile, path)
}

func (m *Manager) addNetwork(n *Network) error {
	m.mux.Lock()
	defer m.mux.Unlock()

	if _, ok := m.networks[n.Name]; ok {
		return errAlreadyExists
	}
	m.networks[n.Name] = n
	return nil
}

func (m *Manager) delNetwork(n *Network) {
	m.mux.Lock()
	delete(m.networks, n.Name)
	m.mux.Unlock()
}

func (m *Manager) getNetwork(netname string) (*Network, bool) {
	m.mux.Lock()
	n, ok := m.networks[netname]
	m.mux.Unlock()

	return n, ok
}

func (m *Manager) forEachNetwork(f func(n *Network)) {
	m.mux.Lock()
	for _, n := range m.networks {
		f(n)
	}
	m.mux.Unlock()
}

func (m *Manager) runNetwork(n *Network) {
	n.Run(m.extIface, func(bn backend.Network) {
		if m.isMultiNetwork() {
			log.Infof("%v: lease acquired: %v", n.Name, bn.Lease().Subnet)

			path := filepath.Join(opts.subnetDir, n.Name) + ".env"
			if err := writeSubnetFile(path, n.Config.Network, m.ipMasq, bn); err != nil {
				log.Warningf("%v failed to write subnet file: %s", n.Name, err)
				return
			}
		} else {
			log.Infof("Lease acquired: %v", bn.Lease().Subnet)

			if err := writeSubnetFile(opts.subnetFile, n.Config.Network, m.ipMasq, bn); err != nil {
				log.Warningf("%v failed to write subnet file: %s", n.Name, err)
				return
			}
			daemon.SdNotify("READY=1")
		}
	})

	m.delNetwork(n)
}

func (m *Manager) watchNetworks() {
	wg := sync.WaitGroup{}
	defer wg.Wait()

	events := make(chan []subnet.Event)
	wg.Add(1)
	go func() {
		subnet.WatchNetworks(m.ctx, m.sm, events)
		wg.Done()
	}()
	// skip over the initial snapshot
	<-events

	for {
		select {
		case <-m.ctx.Done():
			return

		case evtBatch := <-events:
			for _, e := range evtBatch {
				netname := e.Network
				if !m.isNetAllowed(netname) {
					log.Infof("Network %q is not allowed", netname)
					continue
				}

				switch e.Type {
				case subnet.EventAdded:
					n := NewNetwork(m.ctx, m.sm, m.bm, netname, m.ipMasq)
					if err := m.addNetwork(n); err != nil {
						log.Infof("Network %q: %v", netname, err)
						continue
					}

					log.Infof("Network added: %v", netname)

					wg.Add(1)
					go func() {
						m.runNetwork(n)
						wg.Done()
					}()

				case subnet.EventRemoved:
					log.Infof("Network removed: %v", netname)

					n, ok := m.getNetwork(netname)
					if !ok {
						log.Warningf("Network %v unknown; ignoring EventRemoved", netname)
						continue
					}
					n.Cancel()
				}
			}
		}
	}
}

func (m *Manager) Run(ctx context.Context) {
	wg := sync.WaitGroup{}

	if m.isMultiNetwork() {
		for {
			// Try adding initial networks
			result, err := m.sm.WatchNetworks(ctx, nil)
			if err == nil {
				for _, n := range result.Snapshot {
					if m.isNetAllowed(n) {
						m.networks[n] = NewNetwork(ctx, m.sm, m.bm, n, m.ipMasq)
					}
				}
				break
			}

			// Otherwise retry in a few seconds
			log.Warning("Failed to retrieve networks (will retry): %v", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	} else {
		m.networks[""] = NewNetwork(ctx, m.sm, m.bm, "", m.ipMasq)
	}

	// Run existing networks
	m.forEachNetwork(func(n *Network) {
		wg.Add(1)
		go func(n *Network) {
			m.runNetwork(n)
			wg.Done()
		}(n)
	})

	if opts.watchNetworks {
		m.watchNetworks()
	}

	wg.Wait()
	m.bm.Wait()
}
