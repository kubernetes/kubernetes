// Copyright 2019 Google LLC. All Rights Reserved.
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

// Range implements range expression
type Range struct {
	Op       CmpOp
	Register uint32
	FromData []byte
	ToData   []byte
}

func (e *Range) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("range\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Range) marshalData(fam byte) ([]byte, error) {
	var attrs []netlink.Attribute
	var err error
	var rangeFromData, rangeToData []byte

	if e.Register > 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_RANGE_SREG, Data: binaryutil.BigEndian.PutUint32(e.Register)})
	}
	attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_RANGE_OP, Data: binaryutil.BigEndian.PutUint32(uint32(e.Op))})
	if len(e.FromData) > 0 {
		rangeFromData, err = nestedAttr(e.FromData, unix.NFTA_RANGE_FROM_DATA)
		if err != nil {
			return nil, err
		}
	}
	if len(e.ToData) > 0 {
		rangeToData, err = nestedAttr(e.ToData, unix.NFTA_RANGE_TO_DATA)
		if err != nil {
			return nil, err
		}
	}
	data, err := netlink.MarshalAttributes(attrs)
	if err != nil {
		return nil, err
	}
	data = append(data, rangeFromData...)
	data = append(data, rangeToData...)
	return data, nil
}

func (e *Range) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_RANGE_OP:
			e.Op = CmpOp(ad.Uint32())
		case unix.NFTA_RANGE_SREG:
			e.Register = ad.Uint32()
		case unix.NFTA_RANGE_FROM_DATA:
			ad.Do(func(b []byte) error {
				ad, err := netlink.NewAttributeDecoder(b)
				if err != nil {
					return err
				}
				ad.ByteOrder = binary.BigEndian
				if ad.Next() && ad.Type() == unix.NFTA_DATA_VALUE {
					ad.Do(func(b []byte) error {
						e.FromData = b
						return nil
					})
				}
				return ad.Err()
			})
		case unix.NFTA_RANGE_TO_DATA:
			ad.Do(func(b []byte) error {
				ad, err := netlink.NewAttributeDecoder(b)
				if err != nil {
					return err
				}
				ad.ByteOrder = binary.BigEndian
				if ad.Next() && ad.Type() == unix.NFTA_DATA_VALUE {
					ad.Do(func(b []byte) error {
						e.ToData = b
						return nil
					})
				}
				return ad.Err()
			})
		}
	}
	return ad.Err()
}

func nestedAttr(data []byte, attrType uint16) ([]byte, error) {
	ae := netlink.NewAttributeEncoder()
	ae.Do(unix.NLA_F_NESTED|attrType, func() ([]byte, error) {
		nae := netlink.NewAttributeEncoder()
		nae.ByteOrder = binary.BigEndian
		nae.Bytes(unix.NFTA_DATA_VALUE, data)

		return nae.Encode()
	})
	return ae.Encode()
}
