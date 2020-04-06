// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kustomize

import (
	"github.com/spf13/pflag"
)

const (
	flagEnablePluginsName = "enable_alpha_plugins"
	flagEnablePluginsHelp = `enable plugins, an alpha feature.
See https://github.com/kubernetes-sigs/kustomize/blob/master/docs/plugins/README.md
`
)

var (
	flagPluginsEnabledValue = false
)

func addFlagEnablePlugins(set *pflag.FlagSet) {
	set.BoolVar(
		&flagPluginsEnabledValue, flagEnablePluginsName,
		false, flagEnablePluginsHelp)
}

func isFlagEnablePluginsSet() bool {
	return flagPluginsEnabledValue
}
