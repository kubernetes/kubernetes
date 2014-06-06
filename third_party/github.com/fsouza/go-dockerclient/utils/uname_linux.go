// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package utils

import (
	"syscall"
)

type Utsname syscall.Utsname

func uname() (*syscall.Utsname, error) {
	uts := &syscall.Utsname{}

	if err := syscall.Uname(uts); err != nil {
		return nil, err
	}
	return uts, nil
}
