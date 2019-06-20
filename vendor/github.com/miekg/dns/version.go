package dns

import "fmt"

// Version is current version of this library.
var Version = V{1, 1, 4}

// V holds the version of this library.
type V struct {
	Major, Minor, Patch int
}

func (v V) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}
