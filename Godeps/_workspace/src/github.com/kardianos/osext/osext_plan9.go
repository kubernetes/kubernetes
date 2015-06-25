// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package osext

import (
	"os"
	"strconv"
	"syscall"
)

func executable() (string, error) {
	f, err := os.Open("/proc/" + strconv.Itoa(os.Getpid()) + "/text")
	if err != nil {
		return "", err
	}
	defer f.Close()
	return syscall.Fd2path(int(f.Fd()))
}
