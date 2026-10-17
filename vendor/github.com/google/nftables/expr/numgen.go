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

// Numgen defines Numgen expression structure
type Numgen struct {
	Register uint32
	Modulus  uint32
	Type     uint32
	Offset   uint32
}

func (e *Numgen) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("numgen\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Numgen) marshalData(fam byte) ([]byte, error) {
	// Currently only two types are supported, failing if Type is not of two known types
	switch e.Type {
	case unix.NFT_NG_INCREMENTAL:
	case unix.NFT_NG_RANDOM:
	default:
		return nil, fmt.Errorf("unsupported numgen type %d", e.Type)
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_NG_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
		{Type: unix.NFTA_NG_MODULUS, Data: binaryutil.BigEndian.PutUint32(e.Modulus)},
		{Type: unix.NFTA_NG_TYPE, Data: binaryutil.BigEndian.PutUint32(e.Type)},
		{Type: unix.NFTA_NG_OFFSET, Data: binaryutil.BigEndian.PutUint32(e.Offset)},
	})
}

func (e *Numgen) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_NG_DREG:
			e.Register = ad.Uint32()
		case unix.NFTA_NG_MODULUS:
			e.Modulus = ad.Uint32()
		case unix.NFTA_NG_TYPE:
			e.Type = ad.Uint32()
		case unix.NFTA_NG_OFFSET:
			e.Offset = ad.Uint32()
		}
	}
	return ad.Err()
}
