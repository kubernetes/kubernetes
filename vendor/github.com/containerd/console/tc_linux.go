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
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	cmdTcGet = unix.TCGETS
	cmdTcSet = unix.TCSETS
)

// unlockpt unlocks the slave pseudoterminal device corresponding to the master pseudoterminal referred to by f.
// unlockpt should be called before opening the slave side of a pty.
func unlockpt(f *os.File) error {
	var u int32
	// XXX do not use unix.IoctlSetPointerInt here, see commit dbd69c59b81.
	if _, _, err := unix.Syscall(unix.SYS_IOCTL, f.Fd(), unix.TIOCSPTLCK, uintptr(unsafe.Pointer(&u))); err != 0 {
		return err
	}
	return nil
}

// ptsname retrieves the name of the first available pts for the given master.
func ptsname(f *os.File) (string, error) {
	var u uint32
	// XXX do not use unix.IoctlGetInt here, see commit dbd69c59b81.
	if _, _, err := unix.Syscall(unix.SYS_IOCTL, f.Fd(), unix.TIOCGPTN, uintptr(unsafe.Pointer(&u))); err != 0 {
		return "", err
	}
	return fmt.Sprintf("/dev/pts/%d", u), nil
}
