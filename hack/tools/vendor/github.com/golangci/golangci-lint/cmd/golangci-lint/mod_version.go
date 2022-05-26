package main

import (
	"fmt"
	"runtime/debug"
)

//nolint:gochecknoinits
func init() {
	if info, available := debug.ReadBuildInfo(); available {
		if date == "" {
			version = info.Main.Version
			commit = fmt.Sprintf("(unknown, mod sum: %q)", info.Main.Sum)
			date = "(unknown)"
		}
	}
}
