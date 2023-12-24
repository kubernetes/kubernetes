// +build release

package pkcs11

import "fmt"

// Release is current version of the pkcs11 library.
var Release = R{1, 0, 2}

// R holds the version of this library.
type R struct {
	Major, Minor, Patch int
}

func (r R) String() string {
	return fmt.Sprintf("%d.%d.%d", r.Major, r.Minor, r.Patch)
}
