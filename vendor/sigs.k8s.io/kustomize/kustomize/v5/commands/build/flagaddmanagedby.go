// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"os"

	"github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/konfig"
)

const managedByFlag = "enable-managedby-label"

func AddFlagEnableManagedbyLabel(set *pflag.FlagSet) {
	set.BoolVar(
		&theFlags.enable.managedByLabel,
		managedByFlag,
		false,
		`enable adding `+konfig.ManagedbyLabelKey)
}

func isManagedByLabelEnabled() bool {
	if theFlags.enable.managedByLabel {
		return true
	}
	enableLabel, isSet := os.LookupEnv(konfig.EnableManagedbyLabelEnv)
	return isSet && enableLabel == "on"
}
