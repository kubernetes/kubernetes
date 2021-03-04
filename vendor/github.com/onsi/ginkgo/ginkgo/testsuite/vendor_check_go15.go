// +build !go1.6

package testsuite

import (
	"os"
	"path"
)

// "This change will only be enabled if the go command is run with
// GO15VENDOREXPERIMENT=1 in its environment."
// c.f. the vendor-experiment proposal https://goo.gl/2ucMeC
func vendorExperimentCheck(dir string) bool {
	vendorExperiment := os.Getenv("GO15VENDOREXPERIMENT")
	return vendorExperiment == "1" && path.Base(dir) == "vendor"
}
