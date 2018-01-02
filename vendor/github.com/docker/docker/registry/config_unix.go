// +build !windows

package registry

import (
	"github.com/spf13/pflag"
)

var (
	// CertsDir is the directory where certificates are stored
	CertsDir = "/etc/docker/certs.d"
)

// cleanPath is used to ensure that a directory name is valid on the target
// platform. It will be passed in something *similar* to a URL such as
// https:/index.docker.io/v1. Not all platforms support directory names
// which contain those characters (such as : on Windows)
func cleanPath(s string) string {
	return s
}

// installCliPlatformFlags handles any platform specific flags for the service.
func (options *ServiceOptions) installCliPlatformFlags(flags *pflag.FlagSet) {
	flags.BoolVar(&options.V2Only, "disable-legacy-registry", true, "Disable contacting legacy registries")
}
