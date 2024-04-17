//go:build openbsd && !cgo
// +build openbsd,!cgo

/*
   Copyright The containerd Authors.

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

//
// Implementing the functions below requires cgo support.  Non-cgo stubs
// versions are defined below to enable cross-compilation of source code
// that depends on these functions, but the resultant cross-compiled
// binaries cannot actually be used.  If the stub function(s) below are
// actually invoked they will display an error message and cause the
// calling process to exit.
//

package console

import (
	"os"

	"golang.org/x/sys/unix"
)

const (
	cmdTcGet = unix.TIOCGETA
	cmdTcSet = unix.TIOCSETA
)

func ptsname(f *os.File) (string, error) {
	panic("ptsname() support requires cgo.")
}

func unlockpt(f *os.File) error {
	panic("unlockpt() support requires cgo.")
}
