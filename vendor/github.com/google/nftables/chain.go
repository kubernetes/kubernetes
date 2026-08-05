// Copyright 2018 Google LLC. All Rights Reserved.
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

package nftables

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// ChainHook specifies at which step in packet processing the Chain should be
// executed. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Base_chain_hooks
type ChainHook uint32

// Possible ChainHook values.
var (
	ChainHookPrerouting  *ChainHook = ChainHookRef(unix.NF_INET_PRE_ROUTING)
	ChainHookInput       *ChainHook = ChainHookRef(unix.NF_INET_LOCAL_IN)
	ChainHookForward     *ChainHook = ChainHookRef(unix.NF_INET_FORWARD)
	ChainHookOutput      *ChainHook = ChainHookRef(unix.NF_INET_LOCAL_OUT)
	ChainHookPostrouting *ChainHook = ChainHookRef(unix.NF_INET_POST_ROUTING)
	ChainHookIngress     *ChainHook = ChainHookRef(unix.NF_NETDEV_INGRESS)
	ChainHookEgress      *ChainHook = ChainHookRef(unix.NF_NETDEV_EGRESS)
)

// ChainHookRef returns a pointer to a ChainHookRef value.
func ChainHookRef(h ChainHook) *ChainHook {
	return &h
}

// ChainPriority orders the chain relative to Netfilter internal operations. See
// also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Base_chain_priority
type ChainPriority int32

// Possible ChainPriority values.
var ( // from /usr/include/linux/netfilter_ipv4.h
	ChainPriorityFirst            *ChainPriority = ChainPriorityRef(math.MinInt32)
	ChainPriorityConntrackDefrag  *ChainPriority = ChainPriorityRef(-400)
	ChainPriorityRaw              *ChainPriority = ChainPriorityRef(-300)
	ChainPrioritySELinuxFirst     *ChainPriority = ChainPriorityRef(-225)
	ChainPriorityConntrack        *ChainPriority = ChainPriorityRef(-200)
	ChainPriorityMangle           *ChainPriority = ChainPriorityRef(-150)
	ChainPriorityNATDest          *ChainPriority = ChainPriorityRef(-100)
	ChainPriorityFilter           *ChainPriority = ChainPriorityRef(0)
	ChainPrioritySecurity         *ChainPriority = ChainPriorityRef(50)
	ChainPriorityNATSource        *ChainPriority = ChainPriorityRef(100)
	ChainPrioritySELinuxLast      *ChainPriority = ChainPriorityRef(225)
	ChainPriorityConntrackHelper  *ChainPriority = ChainPriorityRef(300)
	ChainPriorityConntrackConfirm *ChainPriority = ChainPriorityRef(math.MaxInt32)
	ChainPriorityLast             *ChainPriority = ChainPriorityRef(math.MaxInt32)
)

// ChainPriorityRef returns a pointer to a ChainPriority value.
func ChainPriorityRef(p ChainPriority) *ChainPriority {
	return &p
}

// ChainType defines what this chain will be used for. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Base_chain_types
type ChainType string

// Possible ChainType values.
const (
	ChainTypeFilter ChainType = "filter"
	ChainTypeRoute  ChainType = "route"
	ChainTypeNAT    ChainType = "nat"
)

// ChainPolicy defines what this chain default policy will be.
type ChainPolicy uint32

// Possible ChainPolicy values.
const (
	ChainPolicyDrop ChainPolicy = iota
	ChainPolicyAccept
)

// A Chain contains Rules. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains
type Chain struct {
	Name     string
	Table    *Table
	Hooknum  *ChainHook
	Priority *ChainPriority
	Type     ChainType
	Policy   *ChainPolicy
	Device   string
}

// AddChain adds the specified Chain. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Adding_base_chains
func (cc *Conn) AddChain(c *Chain) *Chain {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_CHAIN_TABLE, Data: []byte(c.Table.Name + "\x00")},
		{Type: unix.NFTA_CHAIN_NAME, Data: []byte(c.Name + "\x00")},
	})

	if c.Hooknum != nil && c.Priority != nil {
		hookAttr := []netlink.Attribute{
			{Type: unix.NFTA_HOOK_HOOKNUM, Data: binaryutil.BigEndian.PutUint32(uint32(*c.Hooknum))},
			{Type: unix.NFTA_HOOK_PRIORITY, Data: binaryutil.BigEndian.PutUint32(uint32(*c.Priority))},
		}

		if c.Device != "" {
			hookAttr = append(hookAttr, netlink.Attribute{Type: unix.NFTA_HOOK_DEV, Data: []byte(c.Device + "\x00")})
		}

		data = append(data, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NLA_F_NESTED | unix.NFTA_CHAIN_HOOK, Data: cc.marshalAttr(hookAttr)},
		})...)
	}

	if c.Policy != nil {
		data = append(data, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NFTA_CHAIN_POLICY, Data: binaryutil.BigEndian.PutUint32(uint32(*c.Policy))},
		})...)
	}
	if c.Type != "" {
		data = append(data, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NFTA_CHAIN_TYPE, Data: []byte(c.Type + "\x00")},
		})...)
	}
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWCHAIN),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(c.Table.Family), 0), data...),
	})

	return c
}

// DelChain deletes the specified Chain. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Deleting_chains
func (cc *Conn) DelChain(c *Chain) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_CHAIN_TABLE, Data: []byte(c.Table.Name + "\x00")},
		{Type: unix.NFTA_CHAIN_NAME, Data: []byte(c.Name + "\x00")},
	})

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELCHAIN),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(c.Table.Family), 0), data...),
	})
}

// FlushChain removes all rules within the specified Chain. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Configuring_chains#Flushing_chain
func (cc *Conn) FlushChain(c *Chain) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_RULE_TABLE, Data: []byte(c.Table.Name + "\x00")},
		{Type: unix.NFTA_RULE_CHAIN, Data: []byte(c.Name + "\x00")},
	})
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELRULE),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(c.Table.Family), 0), data...),
	})
}

// ListChains returns currently configured chains in the kernel
func (cc *Conn) ListChains() ([]*Chain, error) {
	return cc.ListChainsOfTableFamily(TableFamilyUnspecified)
}

// ListChain returns a single chain configured in the specified table
func (cc *Conn) ListChain(table *Table, chain string) (*Chain, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	attrs := []netlink.Attribute{
		{Type: unix.NFTA_TABLE_NAME, Data: []byte(table.Name + "\x00")},
		{Type: unix.NFTA_CHAIN_NAME, Data: []byte(chain + "\x00")},
	}
	msg := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETCHAIN),
			Flags: netlink.Request,
		},
		Data: append(extraHeader(uint8(table.Family), 0), cc.marshalAttr(attrs)...),
	}

	response, err := conn.Execute(msg)
	if err != nil {
		return nil, fmt.Errorf("conn.Execute failed: %v", err)
	}

	if got, want := len(response), 1; got != want {
		return nil, fmt.Errorf("expected %d response message for chain, got %d", want, got)
	}

	ch, err := chainFromMsg(response[0])
	if err != nil {
		return nil, err
	}

	return ch, nil
}

// ListChainsOfTableFamily returns currently configured chains for the specified
// family in the kernel. It lists all chains ins all tables if family is
// TableFamilyUnspecified.
func (cc *Conn) ListChainsOfTableFamily(family TableFamily) ([]*Chain, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	msg := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETCHAIN),
			Flags: netlink.Request | netlink.Dump,
		},
		Data: extraHeader(uint8(family), 0),
	}

	response, err := conn.Execute(msg)
	if err != nil {
		return nil, err
	}

	var chains []*Chain
	for _, m := range response {
		c, err := chainFromMsg(m)
		if err != nil {
			return nil, err
		}

		chains = append(chains, c)
	}

	return chains, nil
}

func chainFromMsg(msg netlink.Message) (*Chain, error) {
	newChainHeaderType := netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWCHAIN)
	delChainHeaderType := netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELCHAIN)
	if got, want1, want2 := msg.Header.Type, newChainHeaderType, delChainHeaderType; got != want1 && got != want2 {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v or %v", got, want1, want2)
	}

	var c Chain

	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}

	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_CHAIN_NAME:
			c.Name = ad.String()
		case unix.NFTA_TABLE_NAME:
			c.Table = &Table{Name: ad.String()}
			// msg[0] carries TableFamily byte indicating whether it is IPv4, IPv6 or something else
			c.Table.Family = TableFamily(msg.Data[0])
		case unix.NFTA_CHAIN_TYPE:
			c.Type = ChainType(ad.String())
		case unix.NFTA_CHAIN_POLICY:
			policy := ChainPolicy(binaryutil.BigEndian.Uint32(ad.Bytes()))
			c.Policy = &policy
		case unix.NFTA_CHAIN_HOOK:
			ad.Do(func(b []byte) error {
				c.Hooknum, c.Priority, err = hookFromMsg(b)
				return err
			})
		}
	}

	return &c, nil
}

func hookFromMsg(b []byte) (*ChainHook, *ChainPriority, error) {
	ad, err := netlink.NewAttributeDecoder(b)
	if err != nil {
		return nil, nil, err
	}

	ad.ByteOrder = binary.BigEndian

	var hooknum ChainHook
	var prio ChainPriority

	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_HOOK_HOOKNUM:
			hooknum = ChainHook(ad.Uint32())
		case unix.NFTA_HOOK_PRIORITY:
			prio = ChainPriority(ad.Uint32())
		}
	}

	return &hooknum, &prio, nil
}
