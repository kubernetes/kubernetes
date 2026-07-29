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
	"fmt"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type Immediate struct {
	Register uint32
	Data     []byte
}

func (e *Immediate) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("immediate\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Immediate) marshalData(fam byte) ([]byte, error) {
	immData, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_DATA_VALUE, Data: e.Data},
	})
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_IMMEDIATE_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
		{Type: unix.NLA_F_NESTED | unix.NFTA_IMMEDIATE_DATA, Data: immData},
	})
}

func (e *Immediate) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_IMMEDIATE_DREG:
			e.Register = ad.Uint32()
		case unix.NFTA_IMMEDIATE_DATA:
			nestedAD, err := netlink.NewAttributeDecoder(ad.Bytes())
			if err != nil {
				return fmt.Errorf("nested NewAttributeDecoder() failed: %v", err)
			}
			for nestedAD.Next() {
				switch nestedAD.Type() {
				case unix.NFTA_DATA_VALUE:
					e.Data = nestedAD.Bytes()
				}
			}
			if nestedAD.Err() != nil {
				return fmt.Errorf("decoding immediate: %v", nestedAD.Err())
			}
		}
	}
	return ad.Err()
}
