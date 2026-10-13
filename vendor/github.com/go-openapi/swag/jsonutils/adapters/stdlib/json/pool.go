// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
	"github.com/go-openapi/swag/pools"
)

var (
	poolOfAdapters = pools.New[Adapter]()
	poolOfWriters  = pools.NewRedeemable[jwriter]()
	poolOfLexers   = pools.NewRedeemable[jlexer]()
	poolOfReaders  = pools.NewRedeemable[bytesReader]()
)

// BorrowAdapter borrows an [Adapter] from the pool, recycling already allocated instances.
func BorrowAdapter() *Adapter {
	return poolOfAdapters.Borrow()
}

// BorrowAdapterIface borrows a stdlib [Adapter] and converts it directly
// to [ifaces.Adapter].
//
// This is useful to avoid further allocations when translating the concrete type into
// an interface.
func BorrowAdapterIface() ifaces.Adapter {
	return poolOfAdapters.Borrow()
}

// RedeemAdapter redeems an [Adapter] to the pool, so it may be recycled.
func RedeemAdapter(a *Adapter) {
	poolOfAdapters.Redeem(a)
}

func RedeemAdapterIface(a ifaces.Adapter) {
	concrete, ok := a.(*Adapter)
	if ok {
		poolOfAdapters.Redeem(concrete)
	}
}
