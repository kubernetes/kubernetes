// +build experimental

package runconfig

import flag "github.com/docker/docker/pkg/mflag"

type experimentalFlags struct {
	flags map[string]interface{}
}

func attachExperimentalFlags(cmd *flag.FlagSet) *experimentalFlags {
	flags := make(map[string]interface{})
	flags["publish-service"] = cmd.String([]string{"-publish-service"}, "", "Publish this container as a service")
	return &experimentalFlags{flags: flags}
}

func applyExperimentalFlags(exp *experimentalFlags, config *Config, hostConfig *HostConfig) {
	config.PublishService = *(exp.flags["publish-service"]).(*string)
}
