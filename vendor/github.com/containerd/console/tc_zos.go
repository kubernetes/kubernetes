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

package console

import (
	"os"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	cmdTcGet = unix.TCGETS
	cmdTcSet = unix.TCSETS
)

// unlockpt is a no-op on zos.
func unlockpt(_ *os.File) error {
	return nil
}

// ptsname retrieves the name of the first available pts for the given master.
func ptsname(f *os.File) (string, error) {
	return "/dev/ttyp" + strings.TrimPrefix(f.Name(), "/dev/ptyp"), nil
}
