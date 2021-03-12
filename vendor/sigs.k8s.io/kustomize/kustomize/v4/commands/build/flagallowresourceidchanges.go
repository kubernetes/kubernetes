// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"github.com/spf13/pflag"
)

func AddFlagAllowResourceIdChanges(set *pflag.FlagSet) {
	set.BoolVar(
		&theFlags.enable.resourceIdChanges,
		"allow-id-changes",
		false,
		`enable changes to a resourceId`)
}
