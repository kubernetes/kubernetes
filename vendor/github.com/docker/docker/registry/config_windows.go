package registry

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/pflag"
)

// CertsDir is the directory where certificates are stored
var CertsDir = os.Getenv("programdata") + `\docker\certs.d`

// cleanPath is used to ensure that a directory name is valid on the target
// platform. It will be passed in something *similar* to a URL such as
// https:\index.docker.io\v1. Not all platforms support directory names
// which contain those characters (such as : on Windows)
func cleanPath(s string) string {
	return filepath.FromSlash(strings.Replace(s, ":", "", -1))
}

// installCliPlatformFlags handles any platform specific flags for the service.
func (options *ServiceOptions) installCliPlatformFlags(flags *pflag.FlagSet) {
	// No Windows specific flags.
}
