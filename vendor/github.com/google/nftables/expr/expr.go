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

// Package expr provides nftables rule expressions.
package expr

import (
	"encoding/binary"

	"github.com/google/nftables/binaryutil"
	"github.com/google/nftables/internal/parseexprfunc"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

func init() {
	parseExprBytesCommonFunc := func(exprsFromBytesFunc func() ([]Any, error)) ([]interface{}, error) {
		exprs, err := exprsFromBytesFunc()
		if err != nil {
			return nil, err
		}
		result := make([]interface{}, len(exprs))
		for idx, expr := range exprs {
			result[idx] = expr
		}
		return result, nil
	}

	parseexprfunc.ParseExprBytesFromNameFunc = func(fam byte, ad *netlink.AttributeDecoder, exprName string) ([]interface{}, error) {
		return parseExprBytesCommonFunc(func() ([]Any, error) {
			return exprsBytesFromName(fam, ad, exprName)
		})
	}
	parseexprfunc.ParseExprBytesFunc = func(fam byte, ad *netlink.AttributeDecoder) ([]interface{}, error) {
		return parseExprBytesCommonFunc(func() ([]Any, error) {
			return exprsFromBytes(fam, ad)
		})
	}
	parseexprfunc.ParseExprMsgFunc = func(fam byte, b []byte) ([]interface{}, error) {
		ad, err := netlink.NewAttributeDecoder(b)
		if err != nil {
			return nil, err
		}
		ad.ByteOrder = binary.BigEndian
		var exprs []interface{}
		for ad.Next() {
			e, err := parseexprfunc.ParseExprBytesFunc(fam, ad)
			if err != nil {
				return e, err
			}
			exprs = append(exprs, e...)
		}
		return exprs, ad.Err()
	}
}

// Marshal serializes the specified expression into a byte slice.
func Marshal(fam byte, e Any) ([]byte, error) {
	return e.marshal(fam)
}

func MarshalExprData(fam byte, e Any) ([]byte, error) {
	return e.marshalData(fam)
}

// Unmarshal fills an expression from the specified byte slice.
func Unmarshal(fam byte, data []byte, e Any) error {
	return e.unmarshal(fam, data)
}

// exprsBytesFromName parses raw expressions bytes
// based on provided expr name
func exprsBytesFromName(fam byte, ad *netlink.AttributeDecoder, name string) ([]Any, error) {
	var exprs []Any
	e := exprFromName(name)
	ad.Do(func(b []byte) error {
		if err := Unmarshal(fam, b, e); err != nil {
			return err
		}
		exprs = append(exprs, e)
		return nil
	})
	return exprs, ad.Err()
}

// exprsFromBytes parses nested raw expressions bytes
// to construct nftables expressions
func exprsFromBytes(fam byte, ad *netlink.AttributeDecoder) ([]Any, error) {
	var exprs []Any

	ad.Do(func(b []byte) error {
		ad, err := netlink.NewAttributeDecoder(b)
		if err != nil {
			return err
		}
		ad.ByteOrder = binary.BigEndian
		var name string
		for ad.Next() {
			switch ad.Type() {
			case unix.NFTA_EXPR_NAME:
				name = ad.String()
				if name == "notrack" {
					e := &Notrack{}
					exprs = append(exprs, e)
				}
			case unix.NFTA_EXPR_DATA:
				e := exprFromName(name)
				if e == nil {
					// TODO: introduce an opaque expression type so that users know
					// something is here.
					continue // unsupported expression type
				}
				ad.Do(func(b []byte) error {
					if err := Unmarshal(fam, b, e); err != nil {
						return err
					}
					// Verdict expressions are a special-case of immediate expressions, so
					// if the expression is an immediate writing nothing into the verdict
					// register (invalid), re-parse it as a verdict expression.
					if imm, isImmediate := e.(*Immediate); isImmediate && imm.Register == unix.NFT_REG_VERDICT && len(imm.Data) == 0 {
						e = &Verdict{}
						if err := Unmarshal(fam, b, e); err != nil {
							return err
						}
					}
					exprs = append(exprs, e)
					return nil
				})
			}
		}
		return ad.Err()
	})
	return exprs, ad.Err()
}

func exprFromName(name string) Any {
	var e Any
	switch name {
	case "ct":
		e = &Ct{}
	case "range":
		e = &Range{}
	case "meta":
		e = &Meta{}
	case "cmp":
		e = &Cmp{}
	case "counter":
		e = &Counter{}
	case "objref":
		e = &Objref{}
	case "payload":
		e = &Payload{}
	case "lookup":
		e = &Lookup{}
	case "immediate":
		e = &Immediate{}
	case "bitwise":
		e = &Bitwise{}
	case "redir":
		e = &Redir{}
	case "nat":
		e = &NAT{}
	case "limit":
		e = &Limit{}
	case "quota":
		e = &Quota{}
	case "dynset":
		e = &Dynset{}
	case "log":
		e = &Log{}
	case "exthdr":
		e = &Exthdr{}
	case "match":
		e = &Match{}
	case "target":
		e = &Target{}
	case "connlimit":
		e = &Connlimit{}
	case "queue":
		e = &Queue{}
	case "flow_offload":
		e = &FlowOffload{}
	case "reject":
		e = &Reject{}
	case "masq":
		e = &Masq{}
	case "hash":
		e = &Hash{}
	case "cthelper":
		e = &CtHelper{}
	case "synproxy":
		e = &SynProxy{}
	case "ctexpect":
		e = &CtExpect{}
	case "secmark":
		e = &SecMark{}
	case "cttimeout":
		e = &CtTimeout{}
	case "fib":
		e = &Fib{}
	case "numgen":
		e = &Numgen{}
	}
	return e
}

// Any is an interface implemented by any expression type.
type Any interface {
	marshal(fam byte) ([]byte, error)
	marshalData(fam byte) ([]byte, error)
	unmarshal(fam byte, data []byte) error
}

// MetaKey specifies which piece of meta information should be loaded. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Matching_packet_metainformation
type MetaKey uint32

// Possible MetaKey values.
const (
	MetaKeyLEN        MetaKey = unix.NFT_META_LEN
	MetaKeyPROTOCOL   MetaKey = unix.NFT_META_PROTOCOL
	MetaKeyPRIORITY   MetaKey = unix.NFT_META_PRIORITY
	MetaKeyMARK       MetaKey = unix.NFT_META_MARK
	MetaKeyIIF        MetaKey = unix.NFT_META_IIF
	MetaKeyOIF        MetaKey = unix.NFT_META_OIF
	MetaKeyIIFNAME    MetaKey = unix.NFT_META_IIFNAME
	MetaKeyOIFNAME    MetaKey = unix.NFT_META_OIFNAME
	MetaKeyIIFTYPE    MetaKey = unix.NFT_META_IIFTYPE
	MetaKeyOIFTYPE    MetaKey = unix.NFT_META_OIFTYPE
	MetaKeySKUID      MetaKey = unix.NFT_META_SKUID
	MetaKeySKGID      MetaKey = unix.NFT_META_SKGID
	MetaKeyNFTRACE    MetaKey = unix.NFT_META_NFTRACE
	MetaKeyRTCLASSID  MetaKey = unix.NFT_META_RTCLASSID
	MetaKeySECMARK    MetaKey = unix.NFT_META_SECMARK
	MetaKeyNFPROTO    MetaKey = unix.NFT_META_NFPROTO
	MetaKeyL4PROTO    MetaKey = unix.NFT_META_L4PROTO
	MetaKeyBRIIIFNAME MetaKey = unix.NFT_META_BRI_IIFNAME
	MetaKeyBRIOIFNAME MetaKey = unix.NFT_META_BRI_OIFNAME
	MetaKeyPKTTYPE    MetaKey = unix.NFT_META_PKTTYPE
	MetaKeyCPU        MetaKey = unix.NFT_META_CPU
	MetaKeyIIFGROUP   MetaKey = unix.NFT_META_IIFGROUP
	MetaKeyOIFGROUP   MetaKey = unix.NFT_META_OIFGROUP
	MetaKeyCGROUP     MetaKey = unix.NFT_META_CGROUP
	MetaKeyPRANDOM    MetaKey = unix.NFT_META_PRANDOM
)

// Meta loads packet meta information for later comparisons. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Matching_packet_metainformation
type Meta struct {
	Key            MetaKey
	SourceRegister bool
	Register       uint32
}

func (e *Meta) marshal(fam byte) ([]byte, error) {
	exprData, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("meta\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (e *Meta) marshalData(fam byte) ([]byte, error) {
	var regData []byte
	exprData, err := netlink.MarshalAttributes(
		[]netlink.Attribute{
			{Type: unix.NFTA_META_KEY, Data: binaryutil.BigEndian.PutUint32(uint32(e.Key))},
		},
	)
	if err != nil {
		return nil, err
	}
	if e.SourceRegister {
		regData, err = netlink.MarshalAttributes(
			[]netlink.Attribute{
				{Type: unix.NFTA_META_SREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
			},
		)
	} else {
		regData, err = netlink.MarshalAttributes(
			[]netlink.Attribute{
				{Type: unix.NFTA_META_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
			},
		)
	}
	if err != nil {
		return nil, err
	}
	exprData = append(exprData, regData...)
	return exprData, nil
}

func (e *Meta) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_META_SREG:
			e.Register = ad.Uint32()
			e.SourceRegister = true
		case unix.NFTA_META_DREG:
			e.Register = ad.Uint32()
		case unix.NFTA_META_KEY:
			e.Key = MetaKey(ad.Uint32())
		}
	}
	return ad.Err()
}

// Masq (Masquerade) is a special case of SNAT, where the source address is
// automagically set to the address of the output interface. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Performing_Network_Address_Translation_(NAT)#Masquerading
type Masq struct {
	Random      bool
	FullyRandom bool
	Persistent  bool
	ToPorts     bool
	RegProtoMin uint32
	RegProtoMax uint32
}

const (
	// NF_NAT_RANGE_PROTO_RANDOM defines flag for a random masquerade
	NF_NAT_RANGE_PROTO_RANDOM = unix.NF_NAT_RANGE_PROTO_RANDOM
	// NF_NAT_RANGE_PROTO_RANDOM_FULLY defines flag for a fully random masquerade
	NF_NAT_RANGE_PROTO_RANDOM_FULLY = unix.NF_NAT_RANGE_PROTO_RANDOM_FULLY
	// NF_NAT_RANGE_PERSISTENT defines flag for a persistent masquerade
	NF_NAT_RANGE_PERSISTENT = unix.NF_NAT_RANGE_PERSISTENT
	// NF_NAT_RANGE_PREFIX defines flag for a prefix masquerade
	NF_NAT_RANGE_PREFIX = unix.NF_NAT_RANGE_NETMAP
	// NF_NAT_RANGE_PROTO_SPECIFIED defines flag for a specified range
	NF_NAT_RANGE_PROTO_SPECIFIED = unix.NF_NAT_RANGE_PROTO_SPECIFIED
)

func (e *Masq) marshal(fam byte) ([]byte, error) {
	msgData, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("masq\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: msgData},
	})
}

func (e *Masq) marshalData(fam byte) ([]byte, error) {
	msgData := []byte{}
	if !e.ToPorts {
		flags := uint32(0)
		if e.Random {
			flags |= NF_NAT_RANGE_PROTO_RANDOM
		}
		if e.FullyRandom {
			flags |= NF_NAT_RANGE_PROTO_RANDOM_FULLY
		}
		if e.Persistent {
			flags |= NF_NAT_RANGE_PERSISTENT
		}
		if flags != 0 {
			flagsData, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_MASQ_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)}})
			if err != nil {
				return nil, err
			}
			msgData = append(msgData, flagsData...)
		}
	} else {
		regsData, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_MASQ_REG_PROTO_MIN, Data: binaryutil.BigEndian.PutUint32(e.RegProtoMin)}})
		if err != nil {
			return nil, err
		}
		msgData = append(msgData, regsData...)
		if e.RegProtoMax != 0 {
			regsData, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_MASQ_REG_PROTO_MAX, Data: binaryutil.BigEndian.PutUint32(e.RegProtoMax)}})
			if err != nil {
				return nil, err
			}
			msgData = append(msgData, regsData...)
		}
	}
	return msgData, nil
}

func (e *Masq) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_MASQ_REG_PROTO_MIN:
			e.ToPorts = true
			e.RegProtoMin = ad.Uint32()
		case unix.NFTA_MASQ_REG_PROTO_MAX:
			e.RegProtoMax = ad.Uint32()
		case unix.NFTA_MASQ_FLAGS:
			flags := ad.Uint32()
			e.Persistent = (flags & NF_NAT_RANGE_PERSISTENT) != 0
			e.Random = (flags & NF_NAT_RANGE_PROTO_RANDOM) != 0
			e.FullyRandom = (flags & NF_NAT_RANGE_PROTO_RANDOM_FULLY) != 0
		}
	}
	return ad.Err()
}

// CmpOp specifies which type of comparison should be performed.
type CmpOp uint32

// Possible CmpOp values.
const (
	CmpOpEq  CmpOp = unix.NFT_CMP_EQ
	CmpOpNeq CmpOp = unix.NFT_CMP_NEQ
	CmpOpLt  CmpOp = unix.NFT_CMP_LT
	CmpOpLte CmpOp = unix.NFT_CMP_LTE
	CmpOpGt  CmpOp = unix.NFT_CMP_GT
	CmpOpGte CmpOp = unix.NFT_CMP_GTE
)

// Cmp compares a register with the specified data.
type Cmp struct {
	Op       CmpOp
	Register uint32
	Data     []byte
}

func (e *Cmp) marshal(fam byte) ([]byte, error) {
	exprData, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("cmp\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (e *Cmp) marshalData(fam byte) ([]byte, error) {
	cmpData, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_DATA_VALUE, Data: e.Data},
	})
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_CMP_SREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
		{Type: unix.NFTA_CMP_OP, Data: binaryutil.BigEndian.PutUint32(uint32(e.Op))},
		{Type: unix.NLA_F_NESTED | unix.NFTA_CMP_DATA, Data: cmpData},
	})
}

func (e *Cmp) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_CMP_SREG:
			e.Register = ad.Uint32()
		case unix.NFTA_CMP_OP:
			e.Op = CmpOp(ad.Uint32())
		case unix.NFTA_CMP_DATA:
			ad.Do(func(b []byte) error {
				ad, err := netlink.NewAttributeDecoder(b)
				if err != nil {
					return err
				}
				ad.ByteOrder = binary.BigEndian
				if ad.Next() && ad.Type() == unix.NFTA_DATA_VALUE {
					ad.Do(func(b []byte) error {
						e.Data = b
						return nil
					})
				}
				return ad.Err()
			})
		}
	}
	return ad.Err()
}
