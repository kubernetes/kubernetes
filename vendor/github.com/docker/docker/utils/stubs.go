// +build !experimental

package utils

// ExperimentalBuild is a stub which always returns false for
// builds that do not include the "experimental" build tag
func ExperimentalBuild() bool {
	return false
}
