// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"

	"github.com/spf13/pflag"
	flag "github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/krusty"
)

const flagReorderOutputName = "reorder"

func AddFlagReorderOutput(set *pflag.FlagSet) {
	set.StringVar(
		&theFlags.reorderOutput, flagReorderOutputName,
		string(krusty.ReorderOptionLegacy),
		"Reorder the resources just before output. Use '"+string(krusty.ReorderOptionLegacy)+"' to"+
			" apply a legacy reordering (Namespaces first, Webhooks last, etc)."+
			" Use '"+string(krusty.ReorderOptionNone)+"' to suppress a final reordering.")
}

func validateFlagReorderOutput() error {
	switch theFlags.reorderOutput {
	case string(krusty.ReorderOptionNone), string(krusty.ReorderOptionLegacy):
		return nil
	default:
		return fmt.Errorf(
			"illegal flag value --%s %s; legal values: %v",
			flagReorderOutputName, theFlags.reorderOutput,
			[]string{string(krusty.ReorderOptionLegacy), string(krusty.ReorderOptionNone)})
	}
}

func getFlagReorderOutput(flags *flag.FlagSet) krusty.ReorderOption {
	isReorderSet := flags.Changed(flagReorderOutputName)
	if !isReorderSet {
		return krusty.ReorderOptionUnspecified
	}
	switch theFlags.reorderOutput {
	case string(krusty.ReorderOptionNone):
		return krusty.ReorderOptionNone
	case string(krusty.ReorderOptionLegacy):
		return krusty.ReorderOptionLegacy
	default:
		return krusty.ReorderOptionUnspecified
	}
}
