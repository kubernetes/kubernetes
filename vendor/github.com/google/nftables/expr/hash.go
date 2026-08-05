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

type HashType uint32

const (
	HashTypeJenkins HashType = unix.NFT_HASH_JENKINS
	HashTypeSym     HashType = unix.NFT_HASH_SYM
)

// Hash defines type for nftables internal hashing functions
type Hash struct {
	SourceRegister uint32
	DestRegister   uint32
	Length         uint32
	Modulus        uint32
	Seed           uint32
	Offset         uint32
	Type           HashType
}

func (e *Hash) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("hash\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Hash) marshalData(fam byte) ([]byte, error) {
	hashAttrs := []netlink.Attribute{
		{Type: unix.NFTA_HASH_SREG, Data: binaryutil.BigEndian.PutUint32(uint32(e.SourceRegister))},
		{Type: unix.NFTA_HASH_DREG, Data: binaryutil.BigEndian.PutUint32(uint32(e.DestRegister))},
		{Type: unix.NFTA_HASH_LEN, Data: binaryutil.BigEndian.PutUint32(uint32(e.Length))},
		{Type: unix.NFTA_HASH_MODULUS, Data: binaryutil.BigEndian.PutUint32(uint32(e.Modulus))},
	}
	if e.Seed != 0 {
		hashAttrs = append(hashAttrs, netlink.Attribute{
			Type: unix.NFTA_HASH_SEED, Data: binaryutil.BigEndian.PutUint32(uint32(e.Seed)),
		})
	}
	hashAttrs = append(hashAttrs, []netlink.Attribute{
		{Type: unix.NFTA_HASH_OFFSET, Data: binaryutil.BigEndian.PutUint32(uint32(e.Offset))},
		{Type: unix.NFTA_HASH_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(e.Type))},
	}...)
	return netlink.MarshalAttributes(hashAttrs)
}

func (e *Hash) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_HASH_SREG:
			e.SourceRegister = ad.Uint32()
		case unix.NFTA_HASH_DREG:
			e.DestRegister = ad.Uint32()
		case unix.NFTA_HASH_LEN:
			e.Length = ad.Uint32()
		case unix.NFTA_HASH_MODULUS:
			e.Modulus = ad.Uint32()
		case unix.NFTA_HASH_SEED:
			e.Seed = ad.Uint32()
		case unix.NFTA_HASH_OFFSET:
			e.Offset = ad.Uint32()
		case unix.NFTA_HASH_TYPE:
			e.Type = HashType(ad.Uint32())
		}
	}
	return ad.Err()
}
