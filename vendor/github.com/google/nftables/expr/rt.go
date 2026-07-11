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

type RtKey uint32

const (
	RtClassid  RtKey = unix.NFT_RT_CLASSID
	RtNexthop4 RtKey = unix.NFT_RT_NEXTHOP4
	RtNexthop6 RtKey = unix.NFT_RT_NEXTHOP6
	RtTCPMSS   RtKey = unix.NFT_RT_TCPMSS
)

type Rt struct {
	Register uint32
	Key      RtKey
}

func (e *Rt) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("rt\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Rt) marshalData(fam byte) ([]byte, error) {
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_RT_KEY, Data: binaryutil.BigEndian.PutUint32(uint32(e.Key))},
		{Type: unix.NFTA_RT_DREG, Data: binaryutil.BigEndian.PutUint32(e.Register)},
	})
}

func (e *Rt) unmarshal(fam byte, data []byte) error {
	return fmt.Errorf("not yet implemented")
}
