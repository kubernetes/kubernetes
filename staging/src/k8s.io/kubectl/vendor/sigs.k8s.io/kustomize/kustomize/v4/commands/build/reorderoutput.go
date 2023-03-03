// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"

	"github.com/spf13/pflag"
)

//go:generate stringer -type=reorderOutput
type reorderOutput int

const (
	unspecified reorderOutput = iota
	none
	legacy
)

const flagReorderOutputName = "reorder"

func AddFlagReorderOutput(set *pflag.FlagSet) {
	set.StringVar(
		&theFlags.reorderOutput, flagReorderOutputName,
		legacy.String(),
		"Reorder the resources just before output. "+
			"Use '"+legacy.String()+"' to apply a legacy reordering "+
			"(Namespaces first, Webhooks last, etc). "+
			"Use '"+none.String()+"' to suppress a final reordering.")
}

func validateFlagReorderOutput() error {
	switch theFlags.reorderOutput {
	case none.String(), legacy.String():
		return nil
	default:
		return fmt.Errorf(
			"illegal flag value --%s %s; legal values: %v",
			flagReorderOutputName, theFlags.reorderOutput,
			[]string{legacy.String(), none.String()})
	}
}

func getFlagReorderOutput() reorderOutput {
	switch theFlags.reorderOutput {
	case none.String():
		return none
	case legacy.String():
		return legacy
	default:
		return unspecified
	}
}
