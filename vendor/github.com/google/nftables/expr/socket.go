// Copyright 2023 Google LLC. All Rights Reserved.
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

	"golang.org/x/sys/unix"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
)

type Socket struct {
	Key      SocketKey
	Level    uint32
	Register uint32
}

type SocketKey uint32

const (
	// TODO, Once the constants below are available in golang.org/x/sys/unix, switch to use those.
	NFTA_SOCKET_KEY   = 1
	NFTA_SOCKET_DREG  = 2
	NFTA_SOCKET_LEVEL = 3

	NFT_SOCKET_TRANSPARENT = 0
	NFT_SOCKET_MARK        = 1
	NFT_SOCKET_WILDCARD    = 2
	NFT_SOCKET_CGROUPV2    = 3

	SocketKeyTransparent SocketKey = NFT_SOCKET_TRANSPARENT
	SocketKeyMark        SocketKey = NFT_SOCKET_MARK
	SocketKeyWildcard    SocketKey = NFT_SOCKET_WILDCARD
	SocketKeyCgroupv2    SocketKey = NFT_SOCKET_CGROUPV2
)

func (e *Socket) marshal(fam byte) ([]byte, error) {
	exprData, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("socket\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: exprData},
	})
}

func (e *Socket) marshalData(fam byte) ([]byte, error) {
	// NOTE: Socket.Level is only used when Socket.Key == SocketKeyCgroupv2. But `nft` always encoding it. Check link below:
	// http://git.netfilter.org/nftables/tree/src/netlink_linearize.c?id=0583bac241ea18c9d7f61cb20ca04faa1e043b78#n319
	return netlink.MarshalAttributes(
		[]netlink.Attribute{
			{Type: NFTA_SOCKET_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
			{Type: NFTA_SOCKET_KEY, Data: binaryutil.BigEndian.PutUint32(uint32(e.Key))},
			{Type: NFTA_SOCKET_LEVEL, Data: binaryutil.BigEndian.PutUint32(uint32(e.Level))},
		},
	)
}

func (e *Socket) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_SOCKET_DREG:
			e.Register = ad.Uint32()
		case NFTA_SOCKET_KEY:
			e.Key = SocketKey(ad.Uint32())
		case NFTA_SOCKET_LEVEL:
			e.Level = ad.Uint32()
		}
	}
	return ad.Err()
}
