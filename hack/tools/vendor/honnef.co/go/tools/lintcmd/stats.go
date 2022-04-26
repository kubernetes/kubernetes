// +build !aix,!android,!darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris

package lintcmd

import "os"

var infoSignals = []os.Signal{}
