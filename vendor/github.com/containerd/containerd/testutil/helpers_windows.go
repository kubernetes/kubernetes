package testutil

import "testing"

// RequiresRoot does nothing on Windows
func RequiresRoot(t testing.TB) {
}

// RequiresRootM is similar to RequiresRoot but intended to be called from *testing.M.
func RequiresRootM() {
}

// Unmount unmounts a given mountPoint and sets t.Error if it fails
// Does nothing on Windows
func Unmount(t *testing.T, mountPoint string) {
}
