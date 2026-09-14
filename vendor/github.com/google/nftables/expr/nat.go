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

type NATType uint32

// Possible NATType values.
const (
	NATTypeSourceNAT NATType = unix.NFT_NAT_SNAT
	NATTypeDestNAT   NATType = unix.NFT_NAT_DNAT
)

type NAT struct {
	Type        NATType
	Family      uint32 // TODO: typed const
	RegAddrMin  uint32
	RegAddrMax  uint32
	RegProtoMin uint32
	RegProtoMax uint32
	Random      bool
	FullyRandom bool
	Persistent  bool
	Prefix      bool
	Specified   bool
}

// |00048|N-|00001|	|len |flags| type|
// |00008|--|00001|	|len |flags| type|
// | 6e 61 74 00  |	|      data      |	 n a t
// |00036|N-|00002|	|len |flags| type|
// |00008|--|00001|	|len |flags| type| NFTA_NAT_TYPE
// | 00 00 00 01  |	|      data      |  NFT_NAT_DNAT
// |00008|--|00002|	|len |flags| type| NFTA_NAT_FAMILY
// | 00 00 00 02  |	|      data      |   NFPROTO_IPV4
// |00008|--|00003|	|len |flags| type| NFTA_NAT_REG_ADDR_MIN
// | 00 00 00 01  |	|      data      |  reg 1
// |00008|--|00005|	|len |flags| type| NFTA_NAT_REG_PROTO_MIN
// | 00 00 00 02  |	|      data      |  reg 2

func (e *NAT) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("nat\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *NAT) marshalData(fam byte) ([]byte, error) {
	attrs := []netlink.Attribute{
		{Type: unix.NFTA_NAT_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(e.Type))},
		{Type: unix.NFTA_NAT_FAMILY, Data: binaryutil.BigEndian.PutUint32(e.Family)},
	}
	if e.RegAddrMin != 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_NAT_REG_ADDR_MIN, Data: binaryutil.BigEndian.PutUint32(e.RegAddrMin)})
		if e.RegAddrMax != 0 {
			attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_NAT_REG_ADDR_MAX, Data: binaryutil.BigEndian.PutUint32(e.RegAddrMax)})
		}
	}
	if e.RegProtoMin != 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_NAT_REG_PROTO_MIN, Data: binaryutil.BigEndian.PutUint32(e.RegProtoMin)})
		if e.RegProtoMax != 0 {
			attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_NAT_REG_PROTO_MAX, Data: binaryutil.BigEndian.PutUint32(e.RegProtoMax)})
		}
	}
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
	if e.Prefix {
		flags |= NF_NAT_RANGE_PREFIX
	}
	if e.Specified {
		flags |= NF_NAT_RANGE_PROTO_SPECIFIED
	}
	if flags != 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_NAT_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)})
	}

	return netlink.MarshalAttributes(attrs)
}

func (e *NAT) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_NAT_TYPE:
			e.Type = NATType(ad.Uint32())
		case unix.NFTA_NAT_FAMILY:
			e.Family = ad.Uint32()
		case unix.NFTA_NAT_REG_ADDR_MIN:
			e.RegAddrMin = ad.Uint32()
		case unix.NFTA_NAT_REG_ADDR_MAX:
			e.RegAddrMax = ad.Uint32()
		case unix.NFTA_NAT_REG_PROTO_MIN:
			e.RegProtoMin = ad.Uint32()
		case unix.NFTA_NAT_REG_PROTO_MAX:
			e.RegProtoMax = ad.Uint32()
		case unix.NFTA_NAT_FLAGS:
			flags := ad.Uint32()
			e.Persistent = (flags & NF_NAT_RANGE_PERSISTENT) != 0
			e.Random = (flags & NF_NAT_RANGE_PROTO_RANDOM) != 0
			e.FullyRandom = (flags & NF_NAT_RANGE_PROTO_RANDOM_FULLY) != 0
			e.Prefix = (flags & NF_NAT_RANGE_PREFIX) != 0
			e.Specified = (flags & NF_NAT_RANGE_PROTO_SPECIFIED) != 0
		}
	}
	return ad.Err()
}
