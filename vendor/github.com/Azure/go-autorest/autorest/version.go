package autorest

import (
	"fmt"
)

const (
	major        = "7"
	minor        = "0"
	patch        = "0"
	tag          = ""
	semVerFormat = "%s.%s.%s%s"
)

// Version returns the semantic version (see http://semver.org).
func Version() string {
	return fmt.Sprintf(semVerFormat, major, minor, patch, tag)
}
