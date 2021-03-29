// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"

	"github.com/pkg/errors"
)

type errOnlyBuiltinPluginsAllowed struct {
	name string
}

func (e *errOnlyBuiltinPluginsAllowed) Error() string {
	return fmt.Sprintf(
		"external plugins disabled; unable to load external plugin '%s'",
		e.name)
}

func NewErrOnlyBuiltinPluginsAllowed(n string) *errOnlyBuiltinPluginsAllowed {
	return &errOnlyBuiltinPluginsAllowed{name: n}
}

func IsErrOnlyBuiltinPluginsAllowed(err error) bool {
	_, ok := err.(*errOnlyBuiltinPluginsAllowed)
	if ok {
		return true
	}
	_, ok = errors.Cause(err).(*errOnlyBuiltinPluginsAllowed)
	return ok
}
