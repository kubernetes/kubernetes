// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// +build !windows

package capnslog

import (
	"io"
	"os"
	"syscall"
)

// Here's where the opinionation comes in. We need some sensible defaults,
// especially after taking over the log package. Your project (whatever it may
// be) may see things differently. That's okay; there should be no defaults in
// the main package that cannot be controlled or overridden programatically,
// otherwise it's a bug. Doing so is creating your own init_log.go file much
// like this one.

func init() {
	initHijack()

	// Go `log` package uses os.Stderr.
	SetFormatter(NewDefaultFormatter(os.Stderr))
	SetGlobalLogLevel(INFO)
}

func NewDefaultFormatter(out io.Writer) Formatter {
	if syscall.Getppid() == 1 {
		// We're running under init, which may be systemd.
		f, err := NewJournaldFormatter()
		if err == nil {
			return f
		}
	}
	return NewPrettyFormatter(out, false)
}
