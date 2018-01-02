package main

import (
	"os"
	"path/filepath"

	"github.com/docker/docker/daemon/config"
	"github.com/spf13/pflag"
)

var (
	defaultPidFile  string
	defaultDataRoot = filepath.Join(os.Getenv("programdata"), "docker")
)

// installConfigFlags adds flags to the pflag.FlagSet to configure the daemon
func installConfigFlags(conf *config.Config, flags *pflag.FlagSet) {
	// First handle install flags which are consistent cross-platform
	installCommonConfigFlags(conf, flags)

	// Then platform-specific install flags.
	flags.StringVar(&conf.BridgeConfig.FixedCIDR, "fixed-cidr", "", "IPv4 subnet for fixed IPs")
	flags.StringVarP(&conf.BridgeConfig.Iface, "bridge", "b", "", "Attach containers to a virtual switch")
	flags.StringVarP(&conf.SocketGroup, "group", "G", "", "Users or groups that can access the named pipe")
}
