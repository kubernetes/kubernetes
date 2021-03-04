// This work is subject to the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
// license. Its contents can be found at:
// http://creativecommons.org/publicdomain/zero/1.0/

package main

import (
	"fmt"
	"runtime"
)

const (
	AppName         = "go-bindata"
	AppVersionMajor = 3
	AppVersionMinor = 1
)

// revision part of the program version.
// This will be set automatically at build time like so:
//
//     go build -ldflags "-X main.AppVersionRev `date -u +%s`"
var AppVersionRev string

func Version() string {
	if len(AppVersionRev) == 0 {
		AppVersionRev = "1"
	}

	return fmt.Sprintf("%s %d.%d.%s (Go runtime %s).\nCopyright (c) 2010-2013, Jim Teeuwen.",
		AppName, AppVersionMajor, AppVersionMinor, AppVersionRev, runtime.Version())
}
