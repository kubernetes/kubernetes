// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/types"
)

const flagLoadRestrictorName = "load-restrictor"

func AddFlagLoadRestrictor(set *pflag.FlagSet) {
	set.StringVar(
		&theFlags.loadRestrictor,
		flagLoadRestrictorName,
		types.LoadRestrictionsRootOnly.String(),
		"if set to '"+types.LoadRestrictionsNone.String()+
			"', local kustomizations may load files from outside their root. "+
			"This does, however, break the "+
			"relocatability of the kustomization.")
}

func AddFlagLoadRestrictorCompletion(cmd *cobra.Command) error {
	err := cmd.RegisterFlagCompletionFunc(flagLoadRestrictorName, func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		return []string{
			types.LoadRestrictionsNone.String(),
			types.LoadRestrictionsRootOnly.String(),
		}, cobra.ShellCompDirectiveNoFileComp
	})

	if err != nil {
		return fmt.Errorf("unable to add completion for --%s flag: %w", flagLoadRestrictorName, err)
	}

	return nil
}

func validateFlagLoadRestrictor() error {
	switch theFlags.loadRestrictor {
	case types.LoadRestrictionsRootOnly.String(),
		types.LoadRestrictionsNone.String(), "":
		return nil
	default:
		return fmt.Errorf(
			"illegal flag value --%s %s; legal values: %v",
			flagLoadRestrictorName, theFlags.loadRestrictor,
			[]string{types.LoadRestrictionsRootOnly.String(),
				types.LoadRestrictionsNone.String()})
	}
}

func getFlagLoadRestrictorValue() types.LoadRestrictions {
	switch theFlags.loadRestrictor {
	case types.LoadRestrictionsNone.String(), "none":
		return types.LoadRestrictionsNone
	default:
		return types.LoadRestrictionsRootOnly
	}
}
