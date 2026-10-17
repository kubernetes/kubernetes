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

type Reject struct {
	Type uint32
	Code uint8
}

func (e *Reject) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("reject\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Reject) marshalData(fam byte) ([]byte, error) {
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_REJECT_TYPE, Data: binaryutil.BigEndian.PutUint32(e.Type)},
		{Type: unix.NFTA_REJECT_ICMP_CODE, Data: []byte{e.Code}},
	})
}

func (e *Reject) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_REJECT_TYPE:
			e.Type = ad.Uint32()
		case unix.NFTA_REJECT_ICMP_CODE:
			e.Code = ad.Uint8()
		}
	}
	return ad.Err()
}
