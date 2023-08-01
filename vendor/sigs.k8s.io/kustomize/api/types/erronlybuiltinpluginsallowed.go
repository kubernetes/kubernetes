// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"

	"sigs.k8s.io/kustomize/kyaml/errors"
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
	e := &errOnlyBuiltinPluginsAllowed{}
	return errors.As(err, &e)
}
