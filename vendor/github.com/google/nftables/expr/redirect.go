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

type Redir struct {
	RegisterProtoMin uint32
	RegisterProtoMax uint32
	Flags            uint32
}

func (e *Redir) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("redir\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Redir) marshalData(fam byte) ([]byte, error) {
	var attrs []netlink.Attribute
	if e.RegisterProtoMin > 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_REDIR_REG_PROTO_MIN, Data: binaryutil.BigEndian.PutUint32(e.RegisterProtoMin)})
	}
	if e.RegisterProtoMax > 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_REDIR_REG_PROTO_MAX, Data: binaryutil.BigEndian.PutUint32(e.RegisterProtoMax)})
	}
	if e.Flags > 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_REDIR_FLAGS, Data: binaryutil.BigEndian.PutUint32(e.Flags)})
	}

	return netlink.MarshalAttributes(attrs)
}

func (e *Redir) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_REDIR_REG_PROTO_MIN:
			e.RegisterProtoMin = ad.Uint32()
		case unix.NFTA_REDIR_REG_PROTO_MAX:
			e.RegisterProtoMax = ad.Uint32()
		case unix.NFTA_REDIR_FLAGS:
			e.Flags = ad.Uint32()
		}
	}
	return ad.Err()
}
