// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

package svc

import (
	"path/filepath"
	"strings"
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

// IsWindowsService reports whether the process is currently executing
// as a Windows service.
func IsWindowsService() (bool, error) {
	// The below technique looks a bit hairy, but it's actually
	// exactly what the .NET framework does for the similarly named function:
	// https://github.com/dotnet/extensions/blob/f4066026ca06984b07e90e61a6390ac38152ba93/src/Hosting/WindowsServices/src/WindowsServiceHelpers.cs#L26-L31
	// Specifically, it looks up whether the parent process has session ID zero
	// and is called "services".

	var pbi windows.PROCESS_BASIC_INFORMATION
	pbiLen := uint32(unsafe.Sizeof(pbi))
	err := windows.NtQueryInformationProcess(windows.CurrentProcess(), windows.ProcessBasicInformation, unsafe.Pointer(&pbi), pbiLen, &pbiLen)
	if err != nil {
		return false, err
	}
	var psid uint32
	err = windows.ProcessIdToSessionId(uint32(pbi.InheritedFromUniqueProcessId), &psid)
	if err != nil || psid != 0 {
		return false, nil
	}
	pproc, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pbi.InheritedFromUniqueProcessId))
	if err != nil {
		return false, err
	}
	defer windows.CloseHandle(pproc)
	var exeNameBuf [261]uint16
	exeNameLen := uint32(len(exeNameBuf) - 1)
	err = windows.QueryFullProcessImageName(pproc, 0, &exeNameBuf[0], &exeNameLen)
	if err != nil {
		return false, err
	}
	exeName := windows.UTF16ToString(exeNameBuf[:exeNameLen])
	if !strings.EqualFold(filepath.Base(exeName), "services.exe") {
		return false, nil
	}
	system32, err := windows.GetSystemDirectory()
	if err != nil {
		return false, err
	}
	targetExeName := filepath.Join(system32, "services.exe")
	return strings.EqualFold(exeName, targetExeName), nil
}
