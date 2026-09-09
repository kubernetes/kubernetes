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

type Bitwise struct {
	SourceRegister uint32
	DestRegister   uint32
	Len            uint32
	Mask           []byte
	Xor            []byte
}

func (e *Bitwise) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("bitwise\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Bitwise) marshalData(fam byte) ([]byte, error) {
	mask, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_DATA_VALUE, Data: e.Mask},
	})
	if err != nil {
		return nil, err
	}
	xor, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_DATA_VALUE, Data: e.Xor},
	})
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_BITWISE_SREG, Data: binaryutil.BigEndian.PutUint32(e.SourceRegister)},
		{Type: unix.NFTA_BITWISE_DREG, Data: binaryutil.BigEndian.PutUint32(e.DestRegister)},
		{Type: unix.NFTA_BITWISE_LEN, Data: binaryutil.BigEndian.PutUint32(e.Len)},
		{Type: unix.NLA_F_NESTED | unix.NFTA_BITWISE_MASK, Data: mask},
		{Type: unix.NLA_F_NESTED | unix.NFTA_BITWISE_XOR, Data: xor},
	})
}

func (e *Bitwise) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_BITWISE_SREG:
			e.SourceRegister = ad.Uint32()
		case unix.NFTA_BITWISE_DREG:
			e.DestRegister = ad.Uint32()
		case unix.NFTA_BITWISE_LEN:
			e.Len = ad.Uint32()
		case unix.NFTA_BITWISE_MASK:
			// Since NFTA_BITWISE_MASK is nested, it requires additional decoding
			ad.Nested(func(nad *netlink.AttributeDecoder) error {
				for nad.Next() {
					switch nad.Type() {
					case unix.NFTA_DATA_VALUE:
						e.Mask = nad.Bytes()
					}
				}
				return nil
			})
		case unix.NFTA_BITWISE_XOR:
			// Since NFTA_BITWISE_XOR is nested, it requires additional decoding
			ad.Nested(func(nad *netlink.AttributeDecoder) error {
				for nad.Next() {
					switch nad.Type() {
					case unix.NFTA_DATA_VALUE:
						e.Xor = nad.Bytes()
					}
				}
				return nil
			})
		}
	}
	return ad.Err()
}
