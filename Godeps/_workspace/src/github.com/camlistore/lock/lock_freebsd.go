/*
Copyright 2013 The Go Authors

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

package lock

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

func init() {
	lockFn = lockFcntl
}

func lockFcntl(name string) (io.Closer, error) {
	abs, err := filepath.Abs(name)
	if err != nil {
		return nil, err
	}
	lockmu.Lock()
	if locked[abs] {
		lockmu.Unlock()
		return nil, fmt.Errorf("file %q already locked", abs)
	}
	locked[abs] = true
	lockmu.Unlock()

	fi, err := os.Stat(name)
	if err == nil && fi.Size() > 0 {
		return nil, fmt.Errorf("can't Lock file %q: has non-zero size", name)
	}

	f, err := os.Create(name)
	if err != nil {
		return nil, err
	}

	// This type matches C's "struct flock" defined in /usr/include/fcntl.h.
	// TODO: move this into the standard syscall package.
	k := struct {
		Start  int64 /* off_t starting offset */
		Len    int64 /* off_t len = 0 means until end of file */
		Pid    int32 /* pid_t lock owner */
		Type   int16 /* short lock type: read/write, etc. */
		Whence int16 /* short type of l_start */
		Sysid  int32 /* int   remote system id or zero for local */
	}{
		Start:  0,
		Len:    0, // 0 means to lock the entire file.
		Pid:    int32(os.Getpid()),
		Type:   syscall.F_WRLCK,
		Whence: int16(os.SEEK_SET),
		Sysid:  0,
	}

	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), uintptr(syscall.F_SETLK), uintptr(unsafe.Pointer(&k)))
	if errno != 0 {
		f.Close()
		return nil, errno
	}
	return &unlocker{f, abs}, nil
}
