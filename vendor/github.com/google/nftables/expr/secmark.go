// Copyright 2024 Google LLC. All Rights Reserved.
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

	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// From https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=be0bae0ad31b0adb506f96de083f52a2bd0d4fbf#n1338
const (
	NFTA_SECMARK_CTX = 0x01
)

type SecMark struct {
	Ctx string
}

func (e *SecMark) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("secmark\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *SecMark) marshalData(fam byte) ([]byte, error) {
	attrs := []netlink.Attribute{
		{Type: NFTA_SECMARK_CTX, Data: []byte(e.Ctx)},
	}
	return netlink.MarshalAttributes(attrs)
}

func (e *SecMark) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_SECMARK_CTX:
			e.Ctx = ad.String()
		}
	}
	return ad.Err()
}
