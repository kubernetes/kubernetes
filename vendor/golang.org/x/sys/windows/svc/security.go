// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package svc

import (
	"errors"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func allocSid(subAuth0 uint32) (*windows.SID, error) {
	var sid *windows.SID
	err := windows.AllocateAndInitializeSid(&windows.SECURITY_NT_AUTHORITY,
		1, subAuth0, 0, 0, 0, 0, 0, 0, 0, &sid)
	if err != nil {
		return nil, err
	}
	return sid, nil
}

// IsAnInteractiveSession determines if calling process is running interactively.
// It queries the process token for membership in the Interactive group.
// http://stackoverflow.com/questions/2668851/how-do-i-detect-that-my-application-is-running-as-service-or-in-an-interactive-s
//
// Deprecated: Use IsWindowsService instead.
func IsAnInteractiveSession() (bool, error) {
	interSid, err := allocSid(windows.SECURITY_INTERACTIVE_RID)
	if err != nil {
		return false, err
	}
	defer windows.FreeSid(interSid)

	serviceSid, err := allocSid(windows.SECURITY_SERVICE_RID)
	if err != nil {
		return false, err
	}
	defer windows.FreeSid(serviceSid)

	t, err := windows.OpenCurrentProcessToken()
	if err != nil {
		return false, err
	}
	defer t.Close()

	gs, err := t.GetTokenGroups()
	if err != nil {
		return false, err
	}

	for _, g := range gs.AllGroups() {
		if windows.EqualSid(g.Sid, interSid) {
			return true, nil
		}
		if windows.EqualSid(g.Sid, serviceSid) {
			return false, nil
		}
	}
	return false, nil
}

var (
	ntdll                      = windows.NewLazySystemDLL("ntdll.dll")
	_NtQueryInformationProcess = ntdll.NewProc("NtQueryInformationProcess")

	kernel32                    = windows.NewLazySystemDLL("kernel32.dll")
	_QueryFullProcessImageNameA = kernel32.NewProc("QueryFullProcessImageNameA")
)

// IsWindowsService reports whether the process is currently executing
// as a Windows service.
func IsWindowsService() (bool, error) {
	// This code was copied from runtime.isWindowsService function.

	// The below technique looks a bit hairy, but it's actually
	// exactly what the .NET framework does for the similarly named function:
	// https://github.com/dotnet/extensions/blob/f4066026ca06984b07e90e61a6390ac38152ba93/src/Hosting/WindowsServices/src/WindowsServiceHelpers.cs#L26-L31
	// Specifically, it looks up whether the parent process has session ID zero
	// and is called "services".
	const _CURRENT_PROCESS = ^uintptr(0)
	// pbi is a PROCESS_BASIC_INFORMATION struct, where we just care about
	// the 6th pointer inside of it, which contains the pid of the process
	// parent:
	// https://github.com/wine-mirror/wine/blob/42cb7d2ad1caba08de235e6319b9967296b5d554/include/winternl.h#L1294
	var pbi [6]uintptr
	var pbiLen uint32
	r0, _, _ := syscall.Syscall6(_NtQueryInformationProcess.Addr(), 5, _CURRENT_PROCESS, 0, uintptr(unsafe.Pointer(&pbi[0])), uintptr(unsafe.Sizeof(pbi)), uintptr(unsafe.Pointer(&pbiLen)), 0)
	if r0 != 0 {
		return false, errors.New("NtQueryInformationProcess failed: error=" + itoa(int(r0)))
	}
	var psid uint32
	err := windows.ProcessIdToSessionId(uint32(pbi[5]), &psid)
	if err != nil {
		return false, err
	}
	if psid != 0 {
		// parent session id should be 0 for service process
		return false, nil
	}

	pproc, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pbi[5]))
	if err != nil {
		return false, err
	}
	defer windows.CloseHandle(pproc)

	// exeName gets the path to the executable image of the parent process
	var exeName [261]byte
	exeNameLen := uint32(len(exeName) - 1)
	r0, _, e0 := syscall.Syscall6(_QueryFullProcessImageNameA.Addr(), 4, uintptr(pproc), 0, uintptr(unsafe.Pointer(&exeName[0])), uintptr(unsafe.Pointer(&exeNameLen)), 0, 0)
	if r0 == 0 {
		if e0 != 0 {
			return false, e0
		} else {
			return false, syscall.EINVAL
		}
	}
	const (
		servicesLower = "services.exe"
		servicesUpper = "SERVICES.EXE"
	)
	i := int(exeNameLen) - 1
	j := len(servicesLower) - 1
	if i < j {
		return false, nil
	}
	for {
		if j == -1 {
			return i == -1 || exeName[i] == '\\', nil
		}
		if exeName[i] != servicesLower[j] && exeName[i] != servicesUpper[j] {
			return false, nil
		}
		i--
		j--
	}
}

func itoa(val int) string { // do it here rather than with fmt to avoid dependency
	if val < 0 {
		return "-" + itoa(-val)
	}
	var buf [32]byte // big enough for int64
	i := len(buf) - 1
	for val >= 10 {
		buf[i] = byte(val%10 + '0')
		i--
		val /= 10
	}
	buf[i] = byte(val + '0')
	return string(buf[i:])
}
