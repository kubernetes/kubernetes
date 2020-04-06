// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kustomize

import (
	"fmt"

	"github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/types"
)

const (
	flagName = "load_restrictor"
)

var (
	flagLrValue = types.LoadRestrictionsRootOnly.String()
	flagLrHelp  = "if set to '" + types.LoadRestrictionsNone.String() +
		"', local kustomizations may load files from outside their root. " +
		"This does, however, break the relocatability of the kustomization."
)

func addFlagLoadRestrictor(set *pflag.FlagSet) {
	set.StringVar(
		&flagLrValue, flagName,
		types.LoadRestrictionsRootOnly.String(), flagLrHelp)
}

func validateFlagLoadRestrictor() error {
	switch getFlagLoadRestrictorValue() {
	case types.LoadRestrictionsRootOnly, types.LoadRestrictionsNone:
		return nil
	default:
		return fmt.Errorf(
			"illegal flag value --%s %s; legal values: %v",
			flagName, flagLrValue,
			[]string{types.LoadRestrictionsRootOnly.String(), types.LoadRestrictionsNone.String()})
	}
}

func getFlagLoadRestrictorValue() types.LoadRestrictions {
	switch flagLrValue {
	case types.LoadRestrictionsRootOnly.String(), "rootOnly":
		return types.LoadRestrictionsRootOnly
	case types.LoadRestrictionsNone.String(), "none":
		return types.LoadRestrictionsNone
	default:
		return types.LoadRestrictionsUnknown
	}
}
