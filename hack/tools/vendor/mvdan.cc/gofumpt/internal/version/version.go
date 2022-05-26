// Copyright (c) 2020, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package version

import (
	"fmt"
	"os"
	"runtime/debug"
)

var version = "(devel)" // to match the default from runtime/debug

func String() string {
	if testVersion := os.Getenv("GOFUMPT_VERSION_TEST"); testVersion != "" {
		return testVersion
	}
	// don't overwrite the version if it was set by -ldflags=-X
	if info, ok := debug.ReadBuildInfo(); ok && version == "(devel)" {
		mod := &info.Main
		if mod.Replace != nil {
			mod = mod.Replace
		}
		version = mod.Version
	}
	return version
}

func Print() {
	fmt.Println(String())
}
