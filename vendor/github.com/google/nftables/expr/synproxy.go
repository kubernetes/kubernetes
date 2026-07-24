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

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type SynProxy struct {
	Mss       uint16
	Wscale    uint8
	Timestamp bool
	SackPerm  bool
	// Probably not expected to be set by users
	// https://github.com/torvalds/linux/blob/521b1e7f4cf0b05a47995b103596978224b380a8/net/netfilter/nft_synproxy.c#L30-L31
	Ecn bool
	// True when Mss is set to a value or if 0 is an intended value of Mss
	MssValueSet bool
	// True when Wscale is set to a value or if 0 is an intended value of Wscale
	WscaleValueSet bool
}

// From https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=be0bae0ad31b0adb506f96de083f52a2bd0d4fbf#n1723
// Currently not available in golang.org/x/sys/unix
const (
	NFTA_SYNPROXY_MSS    = 0x01
	NFTA_SYNPROXY_WSCALE = 0x02
	NFTA_SYNPROXY_FLAGS  = 0x03
)

// From https://github.com/torvalds/linux/blob/521b1e7f4cf0b05a47995b103596978224b380a8/include/uapi/linux/netfilter/nf_synproxy.h#L7-L15
// Currently not available in golang.org/x/sys/unix
const (
	NF_SYNPROXY_OPT_MSS       = 0x01
	NF_SYNPROXY_OPT_WSCALE    = 0x02
	NF_SYNPROXY_OPT_SACK_PERM = 0x04
	NF_SYNPROXY_OPT_TIMESTAMP = 0x08
	NF_SYNPROXY_OPT_ECN       = 0x10
)

func (e *SynProxy) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("synproxy\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *SynProxy) marshalData(fam byte) ([]byte, error) {
	var flags uint32
	if e.Mss != 0 || e.MssValueSet {
		flags |= NF_SYNPROXY_OPT_MSS
	}
	if e.Wscale != 0 || e.WscaleValueSet {
		flags |= NF_SYNPROXY_OPT_WSCALE
	}
	if e.SackPerm {
		flags |= NF_SYNPROXY_OPT_SACK_PERM
	}
	if e.Timestamp {
		flags |= NF_SYNPROXY_OPT_TIMESTAMP
	}
	if e.Ecn {
		flags |= NF_SYNPROXY_OPT_ECN
	}
	attrs := []netlink.Attribute{
		{Type: NFTA_SYNPROXY_MSS, Data: binaryutil.BigEndian.PutUint16(e.Mss)},
		{Type: NFTA_SYNPROXY_WSCALE, Data: []byte{e.Wscale}},
		{Type: NFTA_SYNPROXY_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)},
	}
	return netlink.MarshalAttributes(attrs)
}

func (e *SynProxy) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case NFTA_SYNPROXY_MSS:
			e.Mss = ad.Uint16()
		case NFTA_SYNPROXY_WSCALE:
			e.Wscale = ad.Uint8()
		case NFTA_SYNPROXY_FLAGS:
			flags := ad.Uint32()
			checkFlag := func(flag uint32) bool {
				return (flags & flag) == flag
			}
			e.MssValueSet = checkFlag(NF_SYNPROXY_OPT_MSS)
			e.WscaleValueSet = checkFlag(NF_SYNPROXY_OPT_WSCALE)
			e.SackPerm = checkFlag(NF_SYNPROXY_OPT_SACK_PERM)
			e.Timestamp = checkFlag(NF_SYNPROXY_OPT_TIMESTAMP)
			e.Ecn = checkFlag(NF_SYNPROXY_OPT_ECN)
		}
	}
	return ad.Err()
}
