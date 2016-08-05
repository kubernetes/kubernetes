// +build go1.6

package testsuite

import (
	"os"
	"path"
)

// in 1.6 the vendor directory became the default go behaviour, so now
// check if its disabled.
func vendorExperimentCheck(dir string) bool {
	vendorExperiment := os.Getenv("GO15VENDOREXPERIMENT")
	return vendorExperiment != "0" && path.Base(dir) == "vendor"
}
