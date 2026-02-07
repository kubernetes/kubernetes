// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	"fmt"
	"reflect"

	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
)

func Register(dispatcher ifaces.Registrar) {
	t := reflect.TypeOf(Adapter{})
	dispatcher.RegisterFor(
		ifaces.RegistryEntry{
			Who:         fmt.Sprintf("%s.%s", t.PkgPath(), t.Name()),
			What:        ifaces.AllCapabilities,
			Constructor: BorrowAdapterIface,
			Support:     support,
		})
}

func support(_ ifaces.Capability, _ any) bool {
	return true
}
