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

// Fib defines fib expression structure
type Fib struct {
	Register       uint32
	ResultOIF      bool
	ResultOIFNAME  bool
	ResultADDRTYPE bool
	FlagSADDR      bool
	FlagDADDR      bool
	FlagMARK       bool
	FlagIIF        bool
	FlagOIF        bool
	FlagPRESENT    bool
}

func (e *Fib) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("fib\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Fib) marshalData(fam byte) ([]byte, error) {
	data := []byte{}
	reg, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_FIB_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
	})
	if err != nil {
		return nil, err
	}
	data = append(data, reg...)
	flags := uint32(0)
	if e.FlagSADDR {
		flags |= unix.NFTA_FIB_F_SADDR
	}
	if e.FlagDADDR {
		flags |= unix.NFTA_FIB_F_DADDR
	}
	if e.FlagMARK {
		flags |= unix.NFTA_FIB_F_MARK
	}
	if e.FlagIIF {
		flags |= unix.NFTA_FIB_F_IIF
	}
	if e.FlagOIF {
		flags |= unix.NFTA_FIB_F_OIF
	}
	if e.FlagPRESENT {
		flags |= unix.NFTA_FIB_F_PRESENT
	}
	if flags != 0 {
		flg, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_FIB_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)},
		})
		if err != nil {
			return nil, err
		}
		data = append(data, flg...)
	}
	results := uint32(0)
	if e.ResultOIF {
		results |= unix.NFT_FIB_RESULT_OIF
	}
	if e.ResultOIFNAME {
		results |= unix.NFT_FIB_RESULT_OIFNAME
	}
	if e.ResultADDRTYPE {
		results |= unix.NFT_FIB_RESULT_ADDRTYPE
	}
	if results != 0 {
		rslt, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_FIB_RESULT, Data: binaryutil.BigEndian.PutUint32(results)},
		})
		if err != nil {
			return nil, err
		}
		data = append(data, rslt...)
	}
	return data, nil
}

func (e *Fib) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_FIB_DREG:
			e.Register = ad.Uint32()
		case unix.NFTA_FIB_RESULT:
			result := ad.Uint32()
			switch result {
			case unix.NFT_FIB_RESULT_OIF:
				e.ResultOIF = true
			case unix.NFT_FIB_RESULT_OIFNAME:
				e.ResultOIFNAME = true
			case unix.NFT_FIB_RESULT_ADDRTYPE:
				e.ResultADDRTYPE = true
			}
		case unix.NFTA_FIB_FLAGS:
			flags := ad.Uint32()
			e.FlagSADDR = (flags & unix.NFTA_FIB_F_SADDR) != 0
			e.FlagDADDR = (flags & unix.NFTA_FIB_F_DADDR) != 0
			e.FlagMARK = (flags & unix.NFTA_FIB_F_MARK) != 0
			e.FlagIIF = (flags & unix.NFTA_FIB_F_IIF) != 0
			e.FlagOIF = (flags & unix.NFTA_FIB_F_OIF) != 0
			e.FlagPRESENT = (flags & unix.NFTA_FIB_F_PRESENT) != 0
		}
	}
	return ad.Err()
}
