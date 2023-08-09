//go:build !go1.12
// +build !go1.12

package shareddefaults

import (
	"os"
	"runtime"
)

func userHomeDir() string {
	if runtime.GOOS == "windows" { // Windows
		return os.Getenv("USERPROFILE")
	}

	// *nix
	return os.Getenv("HOME")
}
