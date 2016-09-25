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

// Package util contains utility functions related to systemd that applications
// can use to check things like whether systemd is running.  Note that some of
// these functions attempt to manually load systemd libraries at runtime rather
// than linking against them.
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
	"io/ioutil"
	"os"
	"strings"
	"syscall"
	"unsafe"

	"github.com/coreos/pkg/dlopen"
)

var libsystemdNames = []string{
	// systemd < 209
	"libsystemd-login.so.0",
	"libsystemd-login.so",

	// systemd >= 209 merged libsystemd-login into libsystemd proper
	"libsystemd.so.0",
	"libsystemd.so",
}

// GetRunningSlice attempts to retrieve the name of the systemd slice in which
// the current process is running.
// This function is a wrapper around the libsystemd C library; if it cannot be
// opened, an error is returned.
func GetRunningSlice() (slice string, err error) {
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

// RunningFromSystemService tries to detect whether the current process has
// been invoked from a system service. The condition for this is whether the
// process is _not_ a user process. User processes are those running in session
// scopes or under per-user `systemd --user` instances.
//
// To avoid false positives on systems without `pam_systemd` (which is
// responsible for creating user sessions), this function also uses a heuristic
// to detect whether it's being invoked from a session leader process. This is
// the case if the current process is executed directly from a service file
// (e.g. with `ExecStart=/this/cmd`). Note that this heuristic will fail if the
// command is instead launched in a subshell or similar so that it is not
// session leader (e.g. `ExecStart=/bin/bash -c "/this/cmd"`)
//
// This function is a wrapper around the libsystemd C library; if this is
// unable to successfully open a handle to the library for any reason (e.g. it
// cannot be found), an errr will be returned
func RunningFromSystemService() (ret bool, err error) {
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
	// ENOENT (systemd <220) or ENXIO (systemd >=220)
	switch {
	case errno >= 0:
		ret = false
	case serrno == syscall.ENOENT, serrno == syscall.ENXIO:
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

// CurrentUnitName attempts to retrieve the name of the systemd system unit
// from which the calling process has been invoked. It wraps the systemd
// `sd_pid_get_unit` call, with the same caveat: for processes not part of a
// systemd system unit, this function will return an error.
func CurrentUnitName() (unit string, err error) {
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

// IsRunningSystemd checks whether the host was booted with systemd as its init
// system. This functions similarly to systemd's `sd_booted(3)`: internally, it
// checks whether /run/systemd/system/ exists and is a directory.
// http://www.freedesktop.org/software/systemd/man/sd_booted.html
func IsRunningSystemd() bool {
	fi, err := os.Lstat("/run/systemd/system")
	if err != nil {
		return false
	}
	return fi.IsDir()
}

// GetMachineID returns a host's 128-bit machine ID as a string. This functions
// similarly to systemd's `sd_id128_get_machine`: internally, it simply reads
// the contents of /etc/machine-id
// http://www.freedesktop.org/software/systemd/man/sd_id128_get_machine.html
func GetMachineID() (string, error) {
	machineID, err := ioutil.ReadFile("/etc/machine-id")
	if err != nil {
		return "", fmt.Errorf("failed to read /etc/machine-id: %v", err)
	}
	return strings.TrimSpace(string(machineID)), nil
}
