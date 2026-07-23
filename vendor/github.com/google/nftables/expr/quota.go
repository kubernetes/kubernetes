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

// Quota defines a threshold against a number of bytes.
type Quota struct {
	Bytes    uint64
	Consumed uint64
	Over     bool
}

func (q *Quota) marshal(fam byte) ([]byte, error) {
	data, err := q.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("quota\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (q *Quota) marshalData(fam byte) ([]byte, error) {
	attrs := []netlink.Attribute{
		{Type: unix.NFTA_QUOTA_BYTES, Data: binaryutil.BigEndian.PutUint64(q.Bytes)},
		{Type: unix.NFTA_QUOTA_CONSUMED, Data: binaryutil.BigEndian.PutUint64(q.Consumed)},
	}

	flags := uint32(0)
	if q.Over {
		flags = unix.NFT_QUOTA_F_INV
	}
	attrs = append(attrs, netlink.Attribute{
		Type: unix.NFTA_QUOTA_FLAGS,
		Data: binaryutil.BigEndian.PutUint32(flags),
	})

	return netlink.MarshalAttributes(attrs)
}

func (q *Quota) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_QUOTA_BYTES:
			q.Bytes = ad.Uint64()
		case unix.NFTA_QUOTA_CONSUMED:
			q.Consumed = ad.Uint64()
		case unix.NFTA_QUOTA_FLAGS:
			q.Over = (ad.Uint32() & unix.NFT_QUOTA_F_INV) != 0
		}
	}
	return ad.Err()
}
