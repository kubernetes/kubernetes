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

const (
	// Per https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n1167
	NFTA_CONNLIMIT_UNSPEC = iota
	NFTA_CONNLIMIT_COUNT
	NFTA_CONNLIMIT_FLAGS
	NFT_CONNLIMIT_F_INV = 1
)

// Per https://git.netfilter.org/libnftnl/tree/src/expr/connlimit.c?id=84d12cfacf8ddd857a09435f3d982ab6250d250c
type Connlimit struct {
	Count uint32
	Flags uint32
}

func (e *Connlimit) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("connlimit\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Connlimit) marshalData(fam byte) ([]byte, error) {
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: NFTA_CONNLIMIT_COUNT, Data: binaryutil.BigEndian.PutUint32(e.Count)},
		{Type: NFTA_CONNLIMIT_FLAGS, Data: binaryutil.BigEndian.PutUint32(e.Flags)},
	})
}

func (e *Connlimit) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_CONNLIMIT_COUNT:
			e.Count = binaryutil.BigEndian.Uint32(ad.Bytes())
		case NFTA_CONNLIMIT_FLAGS:
			e.Flags = binaryutil.BigEndian.Uint32(ad.Bytes())
		}
	}
	return ad.Err()
}
