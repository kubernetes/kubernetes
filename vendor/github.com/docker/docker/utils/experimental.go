// +build experimental

package utils

// ExperimentalBuild is a stub which always returns true for
// builds that include the "experimental" build tag
func ExperimentalBuild() bool {
	return true
}
