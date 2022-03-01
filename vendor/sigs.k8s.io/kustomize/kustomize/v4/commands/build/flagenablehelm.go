// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"github.com/spf13/pflag"
)

// AddFlagEnableHelm adds the --enable-helm flag.
// The helm plugin is builtin, meaning it's
// enabled independently of --enable-alpha-plugins.
func AddFlagEnableHelm(set *pflag.FlagSet) {
	set.BoolVar(
		&theFlags.enable.helm,
		"enable-helm",
		false,
		"Enable use of the Helm chart inflator generator.")
	set.StringVar(
		&theFlags.helmCommand,
		"helm-command",
		"helm", // default
		"helm command (path to executable)")
}
