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

package ip

import (
	"crypto/rand"
	"fmt"
	"net"
	"os"

	"github.com/vishvananda/netlink"
)

func makeVethPair(name, peer string, mtu int) (netlink.Link, error) {
	veth := &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{
			Name:  name,
			Flags: net.FlagUp,
			MTU:   mtu,
		},
		PeerName: peer,
	}
	if err := netlink.LinkAdd(veth); err != nil {
		return nil, err
	}

	return veth, nil
}

func makeVeth(name string, mtu int) (peerName string, veth netlink.Link, err error) {
	for i := 0; i < 10; i++ {
		peerName, err = RandomVethName()
		if err != nil {
			return
		}

		veth, err = makeVethPair(name, peerName, mtu)
		switch {
		case err == nil:
			return

		case os.IsExist(err):
			continue

		default:
			err = fmt.Errorf("failed to make veth pair: %v", err)
			return
		}
	}

	// should really never be hit
	err = fmt.Errorf("failed to find a unique veth name")
	return
}

// RandomVethName returns string "veth" with random prefix (hashed from entropy)
func RandomVethName() (string, error) {
	entropy := make([]byte, 4)
	_, err := rand.Reader.Read(entropy)
	if err != nil {
		return "", fmt.Errorf("failed to generate random veth name: %v", err)
	}

	// NetworkManager (recent versions) will ignore veth devices that start with "veth"
	return fmt.Sprintf("veth%x", entropy), nil
}

// SetupVeth sets up a virtual ethernet link.
// Should be in container netns.
func SetupVeth(contVethName string, mtu int, hostNS *os.File) (hostVeth, contVeth netlink.Link, err error) {
	var hostVethName string
	hostVethName, contVeth, err = makeVeth(contVethName, mtu)
	if err != nil {
		return
	}

	if err = netlink.LinkSetUp(contVeth); err != nil {
		err = fmt.Errorf("failed to set %q up: %v", contVethName, err)
		return
	}

	hostVeth, err = netlink.LinkByName(hostVethName)
	if err != nil {
		err = fmt.Errorf("failed to lookup %q: %v", hostVethName, err)
		return
	}

	if err = netlink.LinkSetUp(hostVeth); err != nil {
		err = fmt.Errorf("failed to set %q up: %v", contVethName, err)
		return
	}

	if err = netlink.LinkSetNsFd(hostVeth, int(hostNS.Fd())); err != nil {
		err = fmt.Errorf("failed to move veth to host netns: %v", err)
		return
	}

	return
}

// DelLinkByName removes an interface link.
func DelLinkByName(ifName string) error {
	iface, err := netlink.LinkByName(ifName)
	if err != nil {
		return fmt.Errorf("failed to lookup %q: %v", ifName, err)
	}

	if err = netlink.LinkDel(iface); err != nil {
		return fmt.Errorf("failed to delete %q: %v", ifName, err)
	}

	return nil
}

// DelLinkByNameAddr remove an interface returns its IP address
// of the specified family
func DelLinkByNameAddr(ifName string, family int) (*net.IPNet, error) {
	iface, err := netlink.LinkByName(ifName)
	if err != nil {
		return nil, fmt.Errorf("failed to lookup %q: %v", ifName, err)
	}

	addrs, err := netlink.AddrList(iface, family)
	if err != nil || len(addrs) == 0 {
		return nil, fmt.Errorf("failed to get IP addresses for %q: %v", ifName, err)
	}

	if err = netlink.LinkDel(iface); err != nil {
		return nil, fmt.Errorf("failed to delete %q: %v", ifName, err)
	}

	return addrs[0].IPNet, nil
}
