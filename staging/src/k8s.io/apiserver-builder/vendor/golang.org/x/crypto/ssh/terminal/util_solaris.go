// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package terminal // import "golang.org/x/crypto/ssh/terminal"

import (
	"golang.org/x/sys/unix"
	"io"
	"syscall"
)

// State contains the state of a terminal.
type State struct {
	termios syscall.Termios
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd int) bool {
	// see: http://src.illumos.org/source/xref/illumos-gate/usr/src/lib/libbc/libc/gen/common/isatty.c
	var termio unix.Termio
	err := unix.IoctlSetTermio(fd, unix.TCGETA, &termio)
	return err == nil
}

// ReadPassword reads a line of input from a terminal without local echo.  This
// is commonly used for inputting passwords and other sensitive data. The slice
// returned does not include the \n.
func ReadPassword(fd int) ([]byte, error) {
	// see also: http://src.illumos.org/source/xref/illumos-gate/usr/src/lib/libast/common/uwin/getpass.c
	val, err := unix.IoctlGetTermios(fd, unix.TCGETS)
	if err != nil {
		return nil, err
	}
	oldState := *val

	newState := oldState
	newState.Lflag &^= syscall.ECHO
	newState.Lflag |= syscall.ICANON | syscall.ISIG
	newState.Iflag |= syscall.ICRNL
	err = unix.IoctlSetTermios(fd, unix.TCSETS, &newState)
	if err != nil {
		return nil, err
	}

	defer unix.IoctlSetTermios(fd, unix.TCSETS, &oldState)

	var buf [16]byte
	var ret []byte
	for {
		n, err := syscall.Read(fd, buf[:])
		if err != nil {
			return nil, err
		}
		if n == 0 {
			if len(ret) == 0 {
				return nil, io.EOF
			}
			break
		}
		if buf[n-1] == '\n' {
			n--
		}
		ret = append(ret, buf[:n]...)
		if n < len(buf) {
			break
		}
	}

	return ret, nil
}
