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
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// This code assembles the verdict structure, as expected by the
// nftables netlink API.
// For further information, consult:
//  - netfilter.h (Linux kernel)
//  - net/netfilter/nf_tables_api.c (Linux kernel)
//  - src/expr/data_reg.c (linbnftnl)

type Verdict struct {
	Kind  VerdictKind
	Chain string
}

type VerdictKind int64

// Verdicts, as per netfilter.h and netfilter/nf_tables.h.
const (
	VerdictReturn VerdictKind = iota - 5
	VerdictGoto
	VerdictJump
	VerdictBreak
	VerdictContinue
	VerdictDrop
	VerdictAccept
	VerdictStolen
	VerdictQueue
	VerdictRepeat
	VerdictStop
)

func (e *Verdict) marshal(fam byte) ([]byte, error) {
	// A verdict is a tree of netlink attributes structured as follows:
	// NFTA_LIST_ELEM | NLA_F_NESTED {
	//   NFTA_EXPR_NAME { "immediate\x00" }
	//   NFTA_EXPR_DATA | NLA_F_NESTED {
	//     NFTA_IMMEDIATE_DREG { NFT_REG_VERDICT }
	//     NFTA_IMMEDIATE_DATA | NLA_F_NESTED {
	//       the verdict code
	//     }
	//   }
	// }
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("immediate\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Verdict) marshalData(fam byte) ([]byte, error) {
	attrs := []netlink.Attribute{
		{Type: unix.NFTA_VERDICT_CODE, Data: binaryutil.BigEndian.PutUint32(uint32(e.Kind))},
	}
	if e.Chain != "" {
		attrs = append(attrs, netlink.Attribute{Type: unix.NFTA_VERDICT_CHAIN, Data: []byte(e.Chain + "\x00")})
	}
	codeData, err := netlink.MarshalAttributes(attrs)
	if err != nil {
		return nil, err
	}

	immData, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NLA_F_NESTED | unix.NFTA_DATA_VERDICT, Data: codeData},
	})
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_IMMEDIATE_DREG, Data: binaryutil.BigEndian.PutUint32(unix.NFT_REG_VERDICT)},
		{Type: unix.NLA_F_NESTED | unix.NFTA_IMMEDIATE_DATA, Data: immData},
	})
}

func (e *Verdict) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_IMMEDIATE_DATA:
			nestedAD, err := netlink.NewAttributeDecoder(ad.Bytes())
			if err != nil {
				return fmt.Errorf("nested NewAttributeDecoder() failed: %v", err)
			}
			for nestedAD.Next() {
				switch nestedAD.Type() {
				case unix.NFTA_DATA_VERDICT:
					e.Kind = VerdictKind(int32(binaryutil.BigEndian.Uint32(nestedAD.Bytes()[4:8])))
					if len(nestedAD.Bytes()) > 12 {
						e.Chain = string(bytes.Trim(nestedAD.Bytes()[12:], "\x00"))
					}
				}
			}
			if nestedAD.Err() != nil {
				return fmt.Errorf("decoding immediate: %v", nestedAD.Err())
			}
		}
	}
	return ad.Err()
}
