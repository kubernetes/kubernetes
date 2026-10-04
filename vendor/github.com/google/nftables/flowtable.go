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

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

const (
	// not in ztypes_linux.go, added here
	// https://cs.opensource.google/go/x/sys/+/c6bc011c:unix/ztypes_linux.go;l=1870-1892
	NFT_MSG_NEWFLOWTABLE = 0x16
	NFT_MSG_GETFLOWTABLE = 0x17
	NFT_MSG_DELFLOWTABLE = 0x18
)

const (
	// not in ztypes_linux.go, added here
	// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n1634
	_ = iota
	NFTA_FLOWTABLE_TABLE
	NFTA_FLOWTABLE_NAME
	NFTA_FLOWTABLE_HOOK
	NFTA_FLOWTABLE_USE
	NFTA_FLOWTABLE_HANDLE
	NFTA_FLOWTABLE_PAD
	NFTA_FLOWTABLE_FLAGS
)

const (
	// not in ztypes_linux.go, added here
	// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n1657
	_ = iota
	NFTA_FLOWTABLE_HOOK_NUM
	NFTA_FLOWTABLE_PRIORITY
	NFTA_FLOWTABLE_DEVS
)

const (
	// not in ztypes_linux.go, added here, used for flowtable device name specification
	// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n1709
	NFTA_DEVICE_NAME = 1
)

type FlowtableFlags uint32

const (
	_ FlowtableFlags = iota
	FlowtableFlagsHWOffload
	FlowtableFlagsCounter
	FlowtableFlagsMask = (FlowtableFlagsHWOffload | FlowtableFlagsCounter)
)

type FlowtableHook uint32

func FlowtableHookRef(h FlowtableHook) *FlowtableHook {
	return &h
}

var (
	// Only ingress is supported
	// https://github.com/torvalds/linux/blob/b72018ab8236c3ae427068adeb94bdd3f20454ec/net/netfilter/nf_tables_api.c#L7378-L7379
	FlowtableHookIngress *FlowtableHook = FlowtableHookRef(unix.NF_NETDEV_INGRESS)
)

type FlowtablePriority int32

func FlowtablePriorityRef(p FlowtablePriority) *FlowtablePriority {
	return &p
}

var (
	// As per man page:
	// The priority can be a signed integer or filter which stands for 0. Addition and subtraction can be used to set relative priority, e.g. filter + 5 equals to 5.
	// https://git.netfilter.org/nftables/tree/doc/nft.txt?id=8c600a843b7c0c1cc275ecc0603bd1fc57773e98#n712
	FlowtablePriorityFilter *FlowtablePriority = FlowtablePriorityRef(0)
)

type Flowtable struct {
	Table    *Table
	Name     string
	Hooknum  *FlowtableHook
	Priority *FlowtablePriority
	Devices  []string
	Use      uint32
	// Bitmask flags, can be HW_OFFLOAD or COUNTER
	// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n1621
	Flags  FlowtableFlags
	Handle uint64
}

func (cc *Conn) AddFlowtable(f *Flowtable) *Flowtable {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	data := cc.marshalAttr([]netlink.Attribute{
		{Type: NFTA_FLOWTABLE_TABLE, Data: []byte(f.Table.Name)},
		{Type: NFTA_FLOWTABLE_NAME, Data: []byte(f.Name)},
		{Type: NFTA_FLOWTABLE_FLAGS, Data: binaryutil.BigEndian.PutUint32(uint32(f.Flags))},
	})

	if f.Hooknum == nil {
		f.Hooknum = FlowtableHookIngress
	}

	if f.Priority == nil {
		f.Priority = FlowtablePriorityFilter
	}

	hookAttr := []netlink.Attribute{
		{Type: NFTA_FLOWTABLE_HOOK_NUM, Data: binaryutil.BigEndian.PutUint32(uint32(*f.Hooknum))},
		{Type: NFTA_FLOWTABLE_PRIORITY, Data: binaryutil.BigEndian.PutUint32(uint32(*f.Priority))},
	}
	if len(f.Devices) > 0 {
		devs := make([]netlink.Attribute, len(f.Devices))
		for i, d := range f.Devices {
			devs[i] = netlink.Attribute{Type: NFTA_DEVICE_NAME, Data: []byte(d)}
		}
		hookAttr = append(hookAttr, netlink.Attribute{
			Type: unix.NLA_F_NESTED | NFTA_FLOWTABLE_DEVS,
			Data: cc.marshalAttr(devs),
		})
	}
	data = append(data, cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NLA_F_NESTED | NFTA_FLOWTABLE_HOOK, Data: cc.marshalAttr(hookAttr)},
	})...)

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | NFT_MSG_NEWFLOWTABLE),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(f.Table.Family), 0), data...),
	})

	return f
}

func (cc *Conn) DelFlowtable(f *Flowtable) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	data := cc.marshalAttr([]netlink.Attribute{
		{Type: NFTA_FLOWTABLE_TABLE, Data: []byte(f.Table.Name)},
		{Type: NFTA_FLOWTABLE_NAME, Data: []byte(f.Name)},
	})

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | NFT_MSG_DELFLOWTABLE),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(f.Table.Family), 0), data...),
	})
}

func (cc *Conn) ListFlowtables(t *Table) ([]*Flowtable, error) {
	reply, err := cc.getFlowtables(t)
	if err != nil {
		return nil, err
	}

	var fts []*Flowtable
	for _, msg := range reply {
		f, err := ftsFromMsg(msg)
		if err != nil {
			return nil, err
		}
		f.Table = t
		fts = append(fts, f)
	}

	return fts, nil
}

func (cc *Conn) getFlowtables(t *Table) ([]netlink.Message, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	attrs := []netlink.Attribute{
		{Type: NFTA_FLOWTABLE_TABLE, Data: []byte(t.Name + "\x00")},
	}
	data, err := netlink.MarshalAttributes(attrs)
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | NFT_MSG_GETFLOWTABLE),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Dump,
		},
		Data: append(extraHeader(uint8(t.Family), 0), data...),
	}

	if _, err := conn.SendMessages([]netlink.Message{message}); err != nil {
		return nil, fmt.Errorf("SendMessages: %v", err)
	}

	reply, err := receiveAckAware(conn, message.Header.Flags)
	if err != nil {
		return nil, fmt.Errorf("receiveAckAware: %v", err)
	}

	return reply, nil
}

func ftsFromMsg(msg netlink.Message) (*Flowtable, error) {
	flowHeaderType := netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | NFT_MSG_NEWFLOWTABLE)
	if got, want := msg.Header.Type, flowHeaderType; got != want {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v", got, want)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian

	var ft Flowtable
	for ad.Next() {
		switch ad.Type() {
		case NFTA_FLOWTABLE_NAME:
			ft.Name = ad.String()
		case NFTA_FLOWTABLE_USE:
			ft.Use = ad.Uint32()
		case NFTA_FLOWTABLE_HANDLE:
			ft.Handle = ad.Uint64()
		case NFTA_FLOWTABLE_FLAGS:
			ft.Flags = FlowtableFlags(ad.Uint32())
		case NFTA_FLOWTABLE_HOOK:
			ad.Do(func(b []byte) error {
				ft.Hooknum, ft.Priority, ft.Devices, err = ftsHookFromMsg(b)
				return err
			})
		}
	}
	return &ft, nil
}

func ftsHookFromMsg(b []byte) (*FlowtableHook, *FlowtablePriority, []string, error) {
	ad, err := netlink.NewAttributeDecoder(b)
	if err != nil {
		return nil, nil, nil, err
	}

	ad.ByteOrder = binary.BigEndian

	var hooknum FlowtableHook
	var prio FlowtablePriority
	var devices []string

	for ad.Next() {
		switch ad.Type() {
		case NFTA_FLOWTABLE_HOOK_NUM:
			hooknum = FlowtableHook(ad.Uint32())
		case NFTA_FLOWTABLE_PRIORITY:
			prio = FlowtablePriority(ad.Uint32())
		case NFTA_FLOWTABLE_DEVS:
			ad.Do(func(b []byte) error {
				devices, err = devsFromMsg(b)
				return err
			})
		}
	}

	return &hooknum, &prio, devices, nil
}

func devsFromMsg(b []byte) ([]string, error) {
	ad, err := netlink.NewAttributeDecoder(b)
	if err != nil {
		return nil, err
	}

	ad.ByteOrder = binary.BigEndian

	devs := make([]string, 0)
	for ad.Next() {
		switch ad.Type() {
		case NFTA_DEVICE_NAME:
			devs = append(devs, ad.String())
		}
	}

	return devs, nil
}
