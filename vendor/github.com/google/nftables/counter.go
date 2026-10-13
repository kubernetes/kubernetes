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

package nftables

import (
	"github.com/google/nftables/expr"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type CounterObj struct {
	Table *Table
	Name  string // e.g. “fwded”

	Bytes   uint64
	Packets uint64
}

func (c *CounterObj) unmarshal(ad *netlink.AttributeDecoder) error {
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_COUNTER_BYTES:
			c.Bytes = ad.Uint64()
		case unix.NFTA_COUNTER_PACKETS:
			c.Packets = ad.Uint64()
		}
	}
	return ad.Err()
}

func (c *CounterObj) data() expr.Any {
	return &expr.Counter{
		Bytes:   c.Bytes,
		Packets: c.Packets,
	}
}

func (c *CounterObj) name() string {
	return c.Name
}
func (c *CounterObj) objType() ObjType {
	return ObjTypeCounter
}

func (c *CounterObj) table() *Table {
	return c.Table
}

func (c *CounterObj) family() TableFamily {
	return c.Table.Family
}
