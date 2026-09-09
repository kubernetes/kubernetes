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
	"fmt"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type ByteorderOp uint32

const (
	ByteorderNtoh ByteorderOp = unix.NFT_BYTEORDER_NTOH
	ByteorderHton ByteorderOp = unix.NFT_BYTEORDER_HTON
)

type Byteorder struct {
	SourceRegister uint32
	DestRegister   uint32
	Op             ByteorderOp
	Len            uint32
	Size           uint32
}

func (e *Byteorder) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("byteorder\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Byteorder) marshalData(fam byte) ([]byte, error) {
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_BYTEORDER_SREG, Data: binaryutil.BigEndian.PutUint32(e.SourceRegister)},
		{Type: unix.NFTA_BYTEORDER_DREG, Data: binaryutil.BigEndian.PutUint32(e.DestRegister)},
		{Type: unix.NFTA_BYTEORDER_OP, Data: binaryutil.BigEndian.PutUint32(uint32(e.Op))},
		{Type: unix.NFTA_BYTEORDER_LEN, Data: binaryutil.BigEndian.PutUint32(e.Len)},
		{Type: unix.NFTA_BYTEORDER_SIZE, Data: binaryutil.BigEndian.PutUint32(e.Size)},
	})
}

func (e *Byteorder) unmarshal(fam byte, data []byte) error {
	return fmt.Errorf("not yet implemented")
}
