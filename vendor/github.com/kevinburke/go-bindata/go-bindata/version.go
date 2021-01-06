// This work is subject to the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
// license. Its contents can be found at:
// http://creativecommons.org/publicdomain/zero/1.0/

package main

import (
	"fmt"
	"runtime"
	"sync"
)

const (
	AppName         = "go-bindata"
	AppVersionMajor = 3
	AppVersionMinor = 22
	AppVersionRev   = 0
)

var vsn, longVsn string
var vsnOnce, longVsnOnce sync.Once

func Version() string {
	vsnOnce.Do(func() {
		vsn = fmt.Sprintf(`go-bindata version %d.%d.%d`, AppVersionMajor, AppVersionMinor, AppVersionRev)
	})
	return vsn
}

func LongVersion() string {
	longVsnOnce.Do(func() {
		longVsn = fmt.Sprintf(`%s %d.%d.%d (Go runtime %s).
Copyright (c) 2010-2015, Jim Teeuwen.
Copyright (c) 2017-2020, Kevin Burke.`, AppName, AppVersionMajor, AppVersionMinor, AppVersionRev, runtime.Version())
	})
	return longVsn
}
