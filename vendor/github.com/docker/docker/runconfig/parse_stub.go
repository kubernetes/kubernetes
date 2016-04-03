// +build !experimental

package runconfig

import flag "github.com/docker/docker/pkg/mflag"

type experimentalFlags struct{}

func attachExperimentalFlags(cmd *flag.FlagSet) *experimentalFlags {
	return nil
}

func applyExperimentalFlags(flags *experimentalFlags, config *Config, hostConfig *HostConfig) {
}
