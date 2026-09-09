// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"github.com/spf13/pflag"
)

func AddFlagEnablePlugins(set *pflag.FlagSet) {
	set.BoolVar(
		&theFlags.enable.plugins,
		"enable-alpha-plugins",
		false,
		"enable kustomize plugins")
}
