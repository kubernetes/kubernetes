package system

import "testing"

func TestHasWin32KSupport(t *testing.T) {
	s := HasWin32KSupport() // make sure this doesn't panic

	t.Logf("win32k: %v", s) // will be different on different platforms -- informative only
}
