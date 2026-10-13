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
	"errors"
	"fmt"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// LimitType represents the type of the limit expression.
type LimitType uint32

// Imported from the nft_limit_type enum in netfilter/nf_tables.h.
const (
	LimitTypePkts     LimitType = unix.NFT_LIMIT_PKTS
	LimitTypePktBytes LimitType = unix.NFT_LIMIT_PKT_BYTES
)

// LimitTime represents the limit unit.
type LimitTime uint64

// Possible limit unit values.
const (
	LimitTimeSecond LimitTime = 1
	LimitTimeMinute LimitTime = 60
	LimitTimeHour   LimitTime = 60 * 60
	LimitTimeDay    LimitTime = 60 * 60 * 24
	LimitTimeWeek   LimitTime = 60 * 60 * 24 * 7
)

func limitTime(value uint64) (LimitTime, error) {
	switch LimitTime(value) {
	case LimitTimeSecond:
		return LimitTimeSecond, nil
	case LimitTimeMinute:
		return LimitTimeMinute, nil
	case LimitTimeHour:
		return LimitTimeHour, nil
	case LimitTimeDay:
		return LimitTimeDay, nil
	case LimitTimeWeek:
		return LimitTimeWeek, nil
	default:
		return 0, fmt.Errorf("expr: invalid limit unit value %d", value)
	}
}

// Limit represents a rate limit expression.
type Limit struct {
	Type  LimitType
	Rate  uint64
	Over  bool
	Unit  LimitTime
	Burst uint32
}

func (l *Limit) marshal(fam byte) ([]byte, error) {
	data, err := l.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("limit\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (l *Limit) marshalData(fam byte) ([]byte, error) {
	var flags uint32
	if l.Over {
		flags = unix.NFT_LIMIT_F_INV
	}
	attrs := []netlink.Attribute{
		{Type: unix.NFTA_LIMIT_RATE, Data: binaryutil.BigEndian.PutUint64(l.Rate)},
		{Type: unix.NFTA_LIMIT_UNIT, Data: binaryutil.BigEndian.PutUint64(uint64(l.Unit))},
		{Type: unix.NFTA_LIMIT_BURST, Data: binaryutil.BigEndian.PutUint32(l.Burst)},
		{Type: unix.NFTA_LIMIT_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(l.Type))},
		{Type: unix.NFTA_LIMIT_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)},
	}

	return netlink.MarshalAttributes(attrs)
}

func (l *Limit) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_LIMIT_RATE:
			l.Rate = ad.Uint64()
		case unix.NFTA_LIMIT_UNIT:
			l.Unit, err = limitTime(ad.Uint64())
			if err != nil {
				return err
			}
		case unix.NFTA_LIMIT_BURST:
			l.Burst = ad.Uint32()
		case unix.NFTA_LIMIT_TYPE:
			l.Type = LimitType(ad.Uint32())
			if l.Type != LimitTypePkts && l.Type != LimitTypePktBytes {
				return fmt.Errorf("expr: invalid limit type %d", l.Type)
			}
		case unix.NFTA_LIMIT_FLAGS:
			l.Over = (ad.Uint32() & unix.NFT_LIMIT_F_INV) != 0
		default:
			return errors.New("expr: unhandled limit netlink attribute")
		}
	}
	return ad.Err()
}
