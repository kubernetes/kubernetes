/*
Copyright 2016 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package osutil

import (
	"bytes"
	"io"
	"os"
)

type crlfWriter struct {
	io.Writer
}

// NewCRLFWriter implements a CR/LF line ending writer used for normalizing
// text for Windows platforms.
func NewCRLFWriter(w io.Writer) io.Writer {
	return crlfWriter{w}
}

func (w crlfWriter) Write(b []byte) (n int, err error) {
	for i, written := 0, 0; ; {
		next := bytes.Index(b[i:], []byte("\n"))
		if next == -1 {
			n, err := w.Writer.Write(b[i:])
			return written + n, err
		}
		next = next + i
		n, err := w.Writer.Write(b[i:next])
		if err != nil {
			return written + n, err
		}
		written += n
		n, err = w.Writer.Write([]byte("\r\n"))
		if err != nil {
			if n > 1 {
				n = 1
			}
			return written + n, err
		}
		written += 1
		i = next + 1
	}
}

// Detects if using systemd as the init system
// Please note that simply reading /proc/1/cmdline can be misleading because
// some installation of various init programs can automatically make /sbin/init
// a symlink or even a renamed version of their main program.
// TODO(dchen1107): realiably detects the init system using on the system:
// systemd, upstart, initd, etc.
func UsingSystemdInitSystem() bool {
	if _, err := os.Stat("/run/systemd/system"); err == nil {
		return true
	}

	return false
}
