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

package expr

import (
	"encoding/binary"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// CtKey specifies which piece of conntrack information should be loaded. See
// also https://wiki.nftables.org/wiki-nftables/index.php/Matching_connection_tracking_stateful_metainformation
type CtKey uint32

// Possible CtKey values.
const (
	CtKeySTATE      CtKey = unix.NFT_CT_STATE
	CtKeyDIRECTION  CtKey = unix.NFT_CT_DIRECTION
	CtKeySTATUS     CtKey = unix.NFT_CT_STATUS
	CtKeyMARK       CtKey = unix.NFT_CT_MARK
	CtKeySECMARK    CtKey = unix.NFT_CT_SECMARK
	CtKeyEXPIRATION CtKey = unix.NFT_CT_EXPIRATION
	CtKeyHELPER     CtKey = unix.NFT_CT_HELPER
	CtKeyL3PROTOCOL CtKey = unix.NFT_CT_L3PROTOCOL
	CtKeySRC        CtKey = unix.NFT_CT_SRC
	CtKeyDST        CtKey = unix.NFT_CT_DST
	CtKeyPROTOCOL   CtKey = unix.NFT_CT_PROTOCOL
	CtKeyPROTOSRC   CtKey = unix.NFT_CT_PROTO_SRC
	CtKeyPROTODST   CtKey = unix.NFT_CT_PROTO_DST
	CtKeyLABELS     CtKey = unix.NFT_CT_LABELS
	CtKeyPKTS       CtKey = unix.NFT_CT_PKTS
	CtKeyBYTES      CtKey = unix.NFT_CT_BYTES
	CtKeyAVGPKT     CtKey = unix.NFT_CT_AVGPKT
	CtKeyZONE       CtKey = unix.NFT_CT_ZONE
	CtKeyEVENTMASK  CtKey = unix.NFT_CT_EVENTMASK

	// https://sources.debian.org/src//nftables/0.9.8-3/src/ct.c/?hl=39#L39
	CtStateBitINVALID     uint32 = 1
	CtStateBitESTABLISHED uint32 = 2
	CtStateBitRELATED     uint32 = 4
	CtStateBitNEW         uint32 = 8
	CtStateBitUNTRACKED   uint32 = 64
)

// Missing ct timeout consts
// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=be0bae0ad31b0adb506f96de083f52a2bd0d4fbf#n1592
const (
	NFTA_CT_TIMEOUT_L3PROTO = 0x01
	NFTA_CT_TIMEOUT_L4PROTO = 0x02
	NFTA_CT_TIMEOUT_DATA    = 0x03
)

type CtStatePolicyTimeout map[uint16]uint32

const (
	// https://git.netfilter.org/libnftnl/tree/src/obj/ct_timeout.c?id=116e95aa7b6358c917de8c69f6f173874030b46b#n24
	CtStateTCPSYNSENT = iota
	CtStateTCPSYNRECV
	CtStateTCPESTABLISHED
	CtStateTCPFINWAIT
	CtStateTCPCLOSEWAIT
	CtStateTCPLASTACK
	CtStateTCPTIMEWAIT
	CtStateTCPCLOSE
	CtStateTCPSYNSENT2
	CtStateTCPRETRANS
	CtStateTCPUNACK
)

// https://git.netfilter.org/libnftnl/tree/src/obj/ct_timeout.c?id=116e95aa7b6358c917de8c69f6f173874030b46b#n38
var CtStateTCPTimeoutDefaults CtStatePolicyTimeout = map[uint16]uint32{
	CtStateTCPSYNSENT:     120,
	CtStateTCPSYNRECV:     60,
	CtStateTCPESTABLISHED: 43200,
	CtStateTCPFINWAIT:     120,
	CtStateTCPCLOSEWAIT:   60,
	CtStateTCPLASTACK:     30,
	CtStateTCPTIMEWAIT:    120,
	CtStateTCPCLOSE:       10,
	CtStateTCPSYNSENT2:    120,
	CtStateTCPRETRANS:     300,
	CtStateTCPUNACK:       300,
}

const (
	// https://git.netfilter.org/libnftnl/tree/src/obj/ct_timeout.c?id=116e95aa7b6358c917de8c69f6f173874030b46b#n57
	CtStateUDPUNREPLIED = iota
	CtStateUDPREPLIED
)

// https://git.netfilter.org/libnftnl/tree/src/obj/ct_timeout.c?id=116e95aa7b6358c917de8c69f6f173874030b46b#n57
var CtStateUDPTimeoutDefaults CtStatePolicyTimeout = map[uint16]uint32{
	CtStateUDPUNREPLIED: 30,
	CtStateUDPREPLIED:   180,
}

// Ct defines type for NFT connection tracking
type Ct struct {
	Register       uint32
	SourceRegister bool
	Key            CtKey
	Direction      uint32
}

func (e *Ct) marshal(fam byte) ([]byte, error) {
	exprData, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("ct\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (e *Ct) marshalData(fam byte) ([]byte, error) {
	var regData []byte
	exprData, err := netlink.MarshalAttributes(
		[]netlink.Attribute{
			{Type: unix.NFTA_CT_KEY, Data: binaryutil.BigEndian.PutUint32(uint32(e.Key))},
		},
	)
	if err != nil {
		return nil, err
	}
	if e.SourceRegister {
		regData, err = netlink.MarshalAttributes(
			[]netlink.Attribute{
				{Type: unix.NFTA_CT_SREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
			},
		)
	} else {
		regData, err = netlink.MarshalAttributes(
			[]netlink.Attribute{
				{Type: unix.NFTA_CT_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
			},
		)
	}
	if err != nil {
		return nil, err
	}
	exprData = append(exprData, regData...)

	switch e.Key {
	case CtKeySRC, CtKeyDST, CtKeyPROTOSRC, CtKeyPROTODST:
		regData, err = netlink.MarshalAttributes(
			[]netlink.Attribute{
				{Type: unix.NFTA_CT_DIRECTION, Data: binaryutil.BigEndian.PutUint32(e.Direction)},
			},
		)
		if err != nil {
			return nil, err
		}
		exprData = append(exprData, regData...)
	}

	return exprData, nil
}

func (e *Ct) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_CT_KEY:
			e.Key = CtKey(ad.Uint32())
		case unix.NFTA_CT_DREG:
			e.Register = ad.Uint32()
		case unix.NFTA_CT_DIRECTION:
			e.Direction = ad.Uint32()
		}
	}
	return ad.Err()
}

type CtHelper struct {
	Name    string
	L3Proto uint16
	L4Proto uint8
}

func (c *CtHelper) marshal(fam byte) ([]byte, error) {
	exprData, err := c.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("cthelper\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (c *CtHelper) marshalData(fam byte) ([]byte, error) {
	exprData := []netlink.Attribute{
		{Type: unix.NFTA_CT_HELPER_NAME, Data: []byte(c.Name)},
	}

	if c.L3Proto != 0 {
		exprData = append(exprData, netlink.Attribute{
			Type: unix.NFTA_CT_HELPER_L3PROTO, Data: binaryutil.BigEndian.PutUint16(c.L3Proto),
		})
	}
	if c.L4Proto != 0 {
		exprData = append(exprData, netlink.Attribute{
			Type: unix.NFTA_CT_HELPER_L4PROTO, Data: []byte{c.L4Proto},
		})
	}

	return netlink.MarshalAttributes(exprData)
}

func (c *CtHelper) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_CT_HELPER_NAME:
			c.Name = ad.String()
		case unix.NFTA_CT_HELPER_L3PROTO:
			c.L3Proto = ad.Uint16()
		case unix.NFTA_CT_HELPER_L4PROTO:
			c.L4Proto = ad.Uint8()
		}
	}
	return ad.Err()
}

// From https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=be0bae0ad31b0adb506f96de083f52a2bd0d4fbf#n1601
// Currently not available in sys/unix
const (
	NFTA_CT_EXPECT_L3PROTO = 0x01
	NFTA_CT_EXPECT_L4PROTO = 0x02
	NFTA_CT_EXPECT_DPORT   = 0x03
	NFTA_CT_EXPECT_TIMEOUT = 0x04
	NFTA_CT_EXPECT_SIZE    = 0x05
)

type CtExpect struct {
	L3Proto uint16
	L4Proto uint8
	DPort   uint16
	Timeout uint32
	Size    uint8
}

func (c *CtExpect) marshal(fam byte) ([]byte, error) {
	exprData, err := c.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("ctexpect\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (c *CtExpect) marshalData(fam byte) ([]byte, error) {
	// all elements except l3proto must be defined
	// per https://git.netfilter.org/nftables/tree/doc/stateful-objects.txt?id=db70959a5ccf2952b218f51c3d529e186a5a43bb#n119
	// from man page: l3proto is derived from the table family by default
	exprData := []netlink.Attribute{
		{Type: NFTA_CT_EXPECT_L4PROTO, Data: []byte{c.L4Proto}},
		{Type: NFTA_CT_EXPECT_DPORT, Data: binaryutil.BigEndian.PutUint16(c.DPort)},
		{Type: NFTA_CT_EXPECT_TIMEOUT, Data: binaryutil.BigEndian.PutUint32(c.Timeout)},
		{Type: NFTA_CT_EXPECT_SIZE, Data: []byte{c.Size}},
	}

	if c.L3Proto != 0 {
		attr := netlink.Attribute{Type: NFTA_CT_EXPECT_L3PROTO, Data: binaryutil.BigEndian.PutUint16(c.L3Proto)}
		exprData = append(exprData, attr)
	}
	return netlink.MarshalAttributes(exprData)
}

func (c *CtExpect) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_CT_EXPECT_L3PROTO:
			c.L3Proto = ad.Uint16()
		case NFTA_CT_EXPECT_L4PROTO:
			c.L4Proto = ad.Uint8()
		case NFTA_CT_EXPECT_DPORT:
			c.DPort = ad.Uint16()
		case NFTA_CT_EXPECT_TIMEOUT:
			c.Timeout = ad.Uint32()
		case NFTA_CT_EXPECT_SIZE:
			c.Size = ad.Uint8()
		}
	}
	return ad.Err()
}

type CtTimeout struct {
	L3Proto uint16
	L4Proto uint8
	Policy  CtStatePolicyTimeout
}

func (c *CtTimeout) marshal(fam byte) ([]byte, error) {
	exprData, err := c.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("cttimeout\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (c *CtTimeout) marshalData(fam byte) ([]byte, error) {
	var policy CtStatePolicyTimeout
	switch c.L4Proto {
	case unix.IPPROTO_UDP:
		policy = CtStateUDPTimeoutDefaults
	default:
		policy = CtStateTCPTimeoutDefaults
	}

	for k, v := range c.Policy {
		policy[k] = v
	}

	var policyAttrs []netlink.Attribute
	for k, v := range policy {
		policyAttrs = append(policyAttrs, netlink.Attribute{Type: k + 1, Data: binaryutil.BigEndian.PutUint32(v)})
	}
	policyData, err := netlink.MarshalAttributes(policyAttrs)
	if err != nil {
		return nil, err
	}

	exprData := []netlink.Attribute{
		{Type: NFTA_CT_TIMEOUT_L3PROTO, Data: binaryutil.BigEndian.PutUint16(c.L3Proto)},
		{Type: NFTA_CT_TIMEOUT_L4PROTO, Data: []byte{c.L4Proto}},
		{Type: unix.NLA_F_NESTED | NFTA_CT_TIMEOUT_DATA, Data: policyData},
	}

	return netlink.MarshalAttributes(exprData)
}

func (c *CtTimeout) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_CT_TIMEOUT_L3PROTO:
			c.L3Proto = ad.Uint16()
		case NFTA_CT_TIMEOUT_L4PROTO:
			c.L4Proto = ad.Uint8()
		case NFTA_CT_TIMEOUT_DATA:
			decoder, err := netlink.NewAttributeDecoder(ad.Bytes())
			decoder.ByteOrder = binary.BigEndian
			if err != nil {
				return err
			}
			for decoder.Next() {
				switch c.L4Proto {
				case unix.IPPROTO_UDP:
					c.Policy = CtStateUDPTimeoutDefaults
				default:
					c.Policy = CtStateTCPTimeoutDefaults
				}
				c.Policy[decoder.Type()-1] = decoder.Uint32()
			}
		}
	}
	return ad.Err()
}
