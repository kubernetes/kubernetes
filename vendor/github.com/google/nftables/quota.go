// Copyright 2023 Google LLC. All Rights Reserved.
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

type QuotaObj struct {
	Table    *Table
	Name     string
	Bytes    uint64
	Consumed uint64
	Over     bool
}

func (q *QuotaObj) unmarshal(ad *netlink.AttributeDecoder) error {
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_QUOTA_BYTES:
			q.Bytes = ad.Uint64()
		case unix.NFTA_QUOTA_CONSUMED:
			q.Consumed = ad.Uint64()
		case unix.NFTA_QUOTA_FLAGS:
			q.Over = (ad.Uint32() & unix.NFT_QUOTA_F_INV) != 0
		}
	}
	return nil
}

func (q *QuotaObj) table() *Table {
	return q.Table
}

func (q *QuotaObj) family() TableFamily {
	return q.Table.Family
}

func (q *QuotaObj) data() expr.Any {
	return &expr.Quota{
		Bytes:    q.Bytes,
		Consumed: q.Consumed,
		Over:     q.Over,
	}
}

func (q *QuotaObj) name() string {
	return q.Name
}

func (q *QuotaObj) objType() ObjType {
	return ObjTypeQuota
}
