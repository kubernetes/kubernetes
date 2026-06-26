//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

// SYSTEM impersonation. The container silo job object created by hcsshim/HCS
// has a security descriptor that grants access only to NT AUTHORITY\SYSTEM and
// the container's own context — not to Administrators. So an elevated-but-not-
// SYSTEM process is denied at open time regardless of the requested access
// mask (and SeDebugPrivilege does not help: it bypasses DACLs only for
// OpenProcess/OpenThread, not job objects). runAsSystem briefly assumes a
// SYSTEM token so the host-side affinity read can open the silo.
//
// Requires the suite to run elevated: Administrators hold SeDebugPrivilege
// (to open a SYSTEM process) and SeImpersonatePrivilege (to set a thread
// token) by default.

const systemSIDString = "S-1-5-18" // NT AUTHORITY\SYSTEM

// runAsSystem runs fn while the current OS thread impersonates SYSTEM, then
// reverts. The goroutine is locked to its thread for the duration because
// SetThreadToken/RevertToSelf are per-thread: locking guarantees fn() runs
// under the impersonated token and that we revert the same thread we changed.
func runAsSystem(fn func() error) (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := enablePrivilege(seDebugName); err != nil {
		return fmt.Errorf("enable SeDebugPrivilege: %w", err)
	}

	sysTok, err := duplicateSystemToken()
	if err != nil {
		return err
	}
	defer sysTok.Close()

	if err := windows.SetThreadToken(nil, sysTok); err != nil {
		return fmt.Errorf("SetThreadToken(SYSTEM): %w", err)
	}
	defer func() {
		if rerr := windows.RevertToSelf(); rerr != nil && err == nil {
			err = fmt.Errorf("RevertToSelf: %w", rerr)
		}
	}()

	return fn()
}

const seDebugName = "SeDebugPrivilege"

// enablePrivilege enables the named privilege on the current process token. It
// is a no-op-equivalent if the privilege is already enabled.
func enablePrivilege(name string) error {
	var tok windows.Token
	if err := windows.OpenProcessToken(windows.CurrentProcess(),
		windows.TOKEN_ADJUST_PRIVILEGES|windows.TOKEN_QUERY, &tok); err != nil {
		return fmt.Errorf("OpenProcessToken: %w", err)
	}
	defer tok.Close()

	namePtr, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return err
	}
	var luid windows.LUID
	if err := windows.LookupPrivilegeValue(nil, namePtr, &luid); err != nil {
		return fmt.Errorf("LookupPrivilegeValue(%s): %w", name, err)
	}
	tp := windows.Tokenprivileges{
		PrivilegeCount: 1,
		Privileges:     [1]windows.LUIDAndAttributes{{Luid: luid, Attributes: windows.SE_PRIVILEGE_ENABLED}},
	}
	// AdjustTokenPrivileges returns success even when the privilege is not held
	// (GetLastError would be ERROR_NOT_ALL_ASSIGNED); in that case the later
	// OpenProcess against a SYSTEM process simply fails and is surfaced there.
	if err := windows.AdjustTokenPrivileges(tok, false, &tp, 0, nil, nil); err != nil {
		return fmt.Errorf("AdjustTokenPrivileges(%s): %w", name, err)
	}
	return nil
}

// duplicateSystemToken walks running processes and duplicates an impersonation
// token from the first one running as SYSTEM whose token can be fully
// duplicated. Returns an error if none can be obtained.
func duplicateSystemToken() (windows.Token, error) {
	systemSID, err := windows.StringToSid(systemSIDString)
	if err != nil {
		return 0, err
	}

	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return 0, fmt.Errorf("CreateToolhelp32Snapshot: %w", err)
	}
	defer windows.CloseHandle(snap)

	var pe windows.ProcessEntry32
	pe.Size = uint32(unsafe.Sizeof(pe))
	for err = windows.Process32First(snap, &pe); err == nil; err = windows.Process32Next(snap, &pe) {
		if pe.ProcessID == 0 {
			continue
		}
		if dup, ok := trySystemTokenFromPID(pe.ProcessID, systemSID); ok {
			return dup, nil
		}
	}
	return 0, fmt.Errorf("could not duplicate a SYSTEM token from any process " +
		"(is the suite running elevated with SeDebugPrivilege?)")
}

// trySystemTokenFromPID duplicates an impersonation token from pid iff that
// process runs as systemSID. It returns ok=false (and no error) on any failure
// so the caller can try the next process: some protected SYSTEM processes
// (e.g. csrss.exe) deny TOKEN_DUPLICATE even with SeDebugPrivilege.
func trySystemTokenFromPID(pid uint32, systemSID *windows.SID) (windows.Token, bool) {
	ph, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return 0, false
	}
	defer windows.CloseHandle(ph)

	var procTok windows.Token
	if err := windows.OpenProcessToken(ph, windows.TOKEN_DUPLICATE|windows.TOKEN_QUERY, &procTok); err != nil {
		return 0, false
	}
	defer procTok.Close()

	tu, err := procTok.GetTokenUser()
	if err != nil || !tu.User.Sid.Equals(systemSID) {
		return 0, false
	}

	var dup windows.Token
	if err := windows.DuplicateTokenEx(procTok,
		windows.TOKEN_QUERY|windows.TOKEN_IMPERSONATE|windows.TOKEN_DUPLICATE,
		nil, windows.SecurityImpersonation, windows.TokenImpersonation, &dup); err != nil {
		return 0, false
	}
	return dup, true
}
