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

type QueueAttribute uint16

type QueueFlag uint16

// Possible QueueAttribute values
const (
	QueueNum   QueueAttribute = unix.NFTA_QUEUE_NUM
	QueueTotal QueueAttribute = unix.NFTA_QUEUE_TOTAL
	QueueFlags QueueAttribute = unix.NFTA_QUEUE_FLAGS

	QueueFlagBypass QueueFlag = unix.NFT_QUEUE_FLAG_BYPASS
	QueueFlagFanout QueueFlag = unix.NFT_QUEUE_FLAG_CPU_FANOUT
	QueueFlagMask   QueueFlag = unix.NFT_QUEUE_FLAG_MASK
)

type Queue struct {
	Num   uint16
	Total uint16
	Flag  QueueFlag
}

func (e *Queue) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("queue\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Queue) marshalData(fam byte) ([]byte, error) {
	if e.Total == 0 {
		e.Total = 1 // The total default value is 1
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_QUEUE_NUM, Data: binaryutil.BigEndian.PutUint16(e.Num)},
		{Type: unix.NFTA_QUEUE_TOTAL, Data: binaryutil.BigEndian.PutUint16(e.Total)},
		{Type: unix.NFTA_QUEUE_FLAGS, Data: binaryutil.BigEndian.PutUint16(uint16(e.Flag))},
	})
}

func (e *Queue) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}
	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_QUEUE_NUM:
			e.Num = ad.Uint16()
		case unix.NFTA_QUEUE_TOTAL:
			e.Total = ad.Uint16()
		case unix.NFTA_QUEUE_FLAGS:
			e.Flag = QueueFlag(ad.Uint16())
		}
	}
	return ad.Err()
}
