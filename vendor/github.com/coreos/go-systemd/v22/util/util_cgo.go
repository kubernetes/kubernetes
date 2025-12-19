// Copyright 2016 CoreOS, Inc.
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

//go:build cgo

package util

// #include <stdlib.h>
// #include <sys/types.h>
// #include <unistd.h>
//
// int
// my_sd_pid_get_owner_uid(void *f, pid_t pid, uid_t *uid)
// {
//   int (*sd_pid_get_owner_uid)(pid_t, uid_t *);
//
//   sd_pid_get_owner_uid = (int (*)(pid_t, uid_t *))f;
//   return sd_pid_get_owner_uid(pid, uid);
// }
//
// int
// my_sd_pid_get_unit(void *f, pid_t pid, char **unit)
// {
//   int (*sd_pid_get_unit)(pid_t, char **);
//
//   sd_pid_get_unit = (int (*)(pid_t, char **))f;
//   return sd_pid_get_unit(pid, unit);
// }
//
// int
// my_sd_pid_get_slice(void *f, pid_t pid, char **slice)
// {
//   int (*sd_pid_get_slice)(pid_t, char **);
//
//   sd_pid_get_slice = (int (*)(pid_t, char **))f;
//   return sd_pid_get_slice(pid, slice);
// }
//
// int
// am_session_leader()
// {
//   return (getsid(0) == getpid());
// }
import "C"

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/coreos/go-systemd/v22/internal/dlopen"
)

var libsystemdNames = []string{
	// systemd < 209
	"libsystemd-login.so.0",
	"libsystemd-login.so",

	// systemd >= 209 merged libsystemd-login into libsystemd proper
	"libsystemd.so.0",
	"libsystemd.so",
}

func getRunningSlice() (slice string, err error) {
	var h *dlopen.LibHandle
	h, err = dlopen.GetHandle(libsystemdNames)
	if err != nil {
		return
	}
	defer func() {
		if err1 := h.Close(); err1 != nil {
			err = err1
		}
	}()

	sd_pid_get_slice, err := h.GetSymbolPointer("sd_pid_get_slice")
	if err != nil {
		return
	}

	var s string
	sl := C.CString(s)
	defer C.free(unsafe.Pointer(sl))

	ret := C.my_sd_pid_get_slice(sd_pid_get_slice, 0, &sl)
	if ret < 0 {
		err = fmt.Errorf("error calling sd_pid_get_slice: %v", syscall.Errno(-ret))
		return
	}

	return C.GoString(sl), nil
}

func runningFromSystemService() (ret bool, err error) {
	var h *dlopen.LibHandle
	h, err = dlopen.GetHandle(libsystemdNames)
	if err != nil {
		return
	}
	defer func() {
		if err1 := h.Close(); err1 != nil {
			err = err1
		}
	}()

	sd_pid_get_owner_uid, err := h.GetSymbolPointer("sd_pid_get_owner_uid")
	if err != nil {
		return
	}

	var uid C.uid_t
	errno := C.my_sd_pid_get_owner_uid(sd_pid_get_owner_uid, 0, &uid)
	serrno := syscall.Errno(-errno)
	// when we're running from a unit file, sd_pid_get_owner_uid returns
	// ENOENT (systemd <220), ENXIO (systemd 220-223), or ENODATA
	// (systemd >=234)
	switch {
	case errno >= 0:
		ret = false
	case serrno == syscall.ENOENT, serrno == syscall.ENXIO, serrno == syscall.ENODATA:
		// Since the implementation of sessions in systemd relies on
		// the `pam_systemd` module, using the sd_pid_get_owner_uid
		// heuristic alone can result in false positives if that module
		// (or PAM itself) is not present or properly configured on the
		// system. As such, we also check if we're the session leader,
		// which should be the case if we're invoked from a unit file,
		// but not if e.g. we're invoked from the command line from a
		// user's login session
		ret = C.am_session_leader() == 1
	default:
		err = fmt.Errorf("error calling sd_pid_get_owner_uid: %v", syscall.Errno(-errno))
	}
	return
}

func currentUnitName() (unit string, err error) {
	var h *dlopen.LibHandle
	h, err = dlopen.GetHandle(libsystemdNames)
	if err != nil {
		return
	}
	defer func() {
		if err1 := h.Close(); err1 != nil {
			err = err1
		}
	}()

	sd_pid_get_unit, err := h.GetSymbolPointer("sd_pid_get_unit")
	if err != nil {
		return
	}

	var s string
	u := C.CString(s)
	defer C.free(unsafe.Pointer(u))

	ret := C.my_sd_pid_get_unit(sd_pid_get_unit, 0, &u)
	if ret < 0 {
		err = fmt.Errorf("error calling sd_pid_get_unit: %v", syscall.Errno(-ret))
		return
	}

	unit = C.GoString(u)
	return
}
