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

package main

import (
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"sync"
	"time"

	"github.com/d2g/dhcp4"
	"github.com/d2g/dhcp4client"
	"github.com/vishvananda/netlink"

	"github.com/appc/cni/pkg/ns"
	"github.com/appc/cni/pkg/types"
)

// RFC 2131 suggests using exponential backoff, starting with 4sec
// and randomized to +/- 1sec
const resendDelay0 = 4 * time.Second
const resendDelayMax = 32 * time.Second

const (
	leaseStateBound = iota
	leaseStateRenewing
	leaseStateRebinding
)

// This implementation uses 1 OS thread per lease. This is because
// all the network operations have to be done in network namespace
// of the interface. This can be improved by switching to the proper
// namespace for network ops and using fewer threads. However, this
// needs to be done carefully as dhcp4client ops are blocking.

type DHCPLease struct {
	clientID      string
	ack           *dhcp4.Packet
	opts          dhcp4.Options
	link          netlink.Link
	renewalTime   time.Time
	rebindingTime time.Time
	expireTime    time.Time
	stop          chan struct{}
	wg            sync.WaitGroup
}

// AcquireLease gets an DHCP lease and then maintains it in the background
// by periodically renewing it. The acquired lease can be released by
// calling DHCPLease.Stop()
func AcquireLease(clientID, netns, ifName string) (*DHCPLease, error) {
	errCh := make(chan error, 1)
	l := &DHCPLease{
		clientID: clientID,
		stop:     make(chan struct{}),
	}

	log.Printf("%v: acquiring lease", clientID)

	l.wg.Add(1)
	go func() {
		errCh <- ns.WithNetNSPath(netns, true, func(_ *os.File) error {
			defer l.wg.Done()

			link, err := netlink.LinkByName(ifName)
			if err != nil {
				return fmt.Errorf("error looking up %q: %v", ifName, err)
			}

			l.link = link

			if err = l.acquire(); err != nil {
				return err
			}

			log.Printf("%v: lease acquired, expiration is %v", l.clientID, l.expireTime)

			errCh <- nil

			l.maintain()
			return nil
		})
	}()

	if err := <-errCh; err != nil {
		return nil, err
	}

	return l, nil
}

// Stop terminates the background task that maintains the lease
// and issues a DHCP Release
func (l *DHCPLease) Stop() {
	close(l.stop)
	l.wg.Wait()
}

func (l *DHCPLease) acquire() error {
	c, err := newDHCPClient(l.link)
	if err != nil {
		return err
	}
	defer c.Close()

	pkt, err := backoffRetry(func() (*dhcp4.Packet, error) {
		ok, ack, err := c.Request()
		switch {
		case err != nil:
			return nil, err
		case !ok:
			return nil, fmt.Errorf("DHCP server NACK'd own offer")
		default:
			return &ack, nil
		}
	})
	if err != nil {
		return err
	}

	return l.commit(pkt)
}

func (l *DHCPLease) commit(ack *dhcp4.Packet) error {
	opts := ack.ParseOptions()

	leaseTime, err := parseLeaseTime(opts)
	if err != nil {
		return err
	}

	rebindingTime, err := parseRebindingTime(opts)
	if err != nil || rebindingTime > leaseTime {
		// Per RFC 2131 Section 4.4.5, it should default to 85% of lease time
		rebindingTime = leaseTime * 85 / 100
	}

	renewalTime, err := parseRenewalTime(opts)
	if err != nil || renewalTime > rebindingTime {
		// Per RFC 2131 Section 4.4.5, it should default to 50% of lease time
		renewalTime = leaseTime / 2
	}

	now := time.Now()
	l.expireTime = now.Add(leaseTime)
	l.renewalTime = now.Add(renewalTime)
	l.rebindingTime = now.Add(rebindingTime)
	l.ack = ack
	l.opts = opts

	return nil
}

func (l *DHCPLease) maintain() {
	state := leaseStateBound

	for {
		var sleepDur time.Duration

		switch state {
		case leaseStateBound:
			sleepDur = l.renewalTime.Sub(time.Now())
			if sleepDur <= 0 {
				log.Printf("%v: renewing lease", l.clientID)
				state = leaseStateRenewing
				continue
			}

		case leaseStateRenewing:
			if err := l.renew(); err != nil {
				log.Printf("%v: %v", l.clientID, err)

				if time.Now().After(l.rebindingTime) {
					log.Printf("%v: renawal time expired, rebinding", l.clientID)
					state = leaseStateRebinding
				}
			} else {
				log.Printf("%v: lease renewed, expiration is %v", l.clientID, l.expireTime)
				state = leaseStateBound
			}

		case leaseStateRebinding:
			if err := l.acquire(); err != nil {
				log.Printf("%v: %v", l.clientID, err)

				if time.Now().After(l.expireTime) {
					log.Printf("%v: lease expired, bringing interface DOWN", l.clientID)
					l.downIface()
					return
				}
			} else {
				log.Printf("%v: lease rebound, expiration is %v", l.clientID, l.expireTime)
				state = leaseStateBound
			}
		}

		select {
		case <-time.After(sleepDur):

		case <-l.stop:
			if err := l.release(); err != nil {
				log.Printf("%v: failed to release DHCP lease: %v", l.clientID, err)
			}
			return
		}
	}
}

func (l *DHCPLease) downIface() {
	if err := netlink.LinkSetDown(l.link); err != nil {
		log.Printf("%v: failed to bring %v interface DOWN: %v", l.clientID, l.link.Attrs().Name, err)
	}
}

func (l *DHCPLease) renew() error {
	c, err := newDHCPClient(l.link)
	if err != nil {
		return err
	}
	defer c.Close()

	pkt, err := backoffRetry(func() (*dhcp4.Packet, error) {
		ok, ack, err := c.Renew(*l.ack)
		switch {
		case err != nil:
			return nil, err
		case !ok:
			return nil, fmt.Errorf("DHCP server did not renew lease")
		default:
			return &ack, nil
		}
	})
	if err != nil {
		return err
	}

	l.commit(pkt)
	return nil
}

func (l *DHCPLease) release() error {
	log.Printf("%v: releasing lease", l.clientID)

	c, err := newDHCPClient(l.link)
	if err != nil {
		return err
	}
	defer c.Close()

	if err = c.Release(*l.ack); err != nil {
		return fmt.Errorf("failed to send DHCPRELEASE")
	}

	return nil
}

func (l *DHCPLease) IPNet() (*net.IPNet, error) {
	mask := parseSubnetMask(l.opts)
	if mask == nil {
		return nil, fmt.Errorf("DHCP option Subnet Mask not found in DHCPACK")
	}

	return &net.IPNet{
		IP:   l.ack.YIAddr(),
		Mask: mask,
	}, nil
}

func (l *DHCPLease) Gateway() net.IP {
	return parseRouter(l.opts)
}

func (l *DHCPLease) Routes() []types.Route {
	routes := parseRoutes(l.opts)
	return append(routes, parseCIDRRoutes(l.opts)...)
}

// jitter returns a random value within [-span, span) range
func jitter(span time.Duration) time.Duration {
	return time.Duration(float64(span) * (2.0*rand.Float64() - 1.0))
}

func backoffRetry(f func() (*dhcp4.Packet, error)) (*dhcp4.Packet, error) {
	var baseDelay time.Duration = resendDelay0

	for i := 0; i < resendCount; i++ {
		pkt, err := f()
		if err == nil {
			return pkt, nil
		}

		log.Print(err)

		time.Sleep(baseDelay + jitter(time.Second))

		if baseDelay < resendDelayMax {
			baseDelay *= 2
		}
	}

	return nil, errNoMoreTries
}

func newDHCPClient(link netlink.Link) (*dhcp4client.Client, error) {
	pktsock, err := dhcp4client.NewPacketSock(link.Attrs().Index)
	if err != nil {
		return nil, err
	}

	return dhcp4client.New(
		dhcp4client.HardwareAddr(link.Attrs().HardwareAddr),
		dhcp4client.Timeout(5*time.Second),
		dhcp4client.Broadcast(false),
		dhcp4client.Connection(pktsock),
	)
}
