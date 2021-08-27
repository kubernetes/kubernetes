//go:build windows
// +build windows

/*
Copyright 2023 The Kubernetes Authors.

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

package util

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	// https://docs.microsoft.com/en-us/windows/win32/fileio/file-access-rights-constants
	// read = read data | read attributes
	READ_PERMISSIONS = 0x0001 | 0x0080

	// write = write data | append data | write attributes | write EA
	WRITE_PERMISSIONS = 0x0002 | 0x0004 | 0x0100 | 0x0010

	// execute = read data | file execute
	EXECUTE_PERMISSIONS = 0x0001 | 0x0020

	// Accounts
	EVERYONE = "Everyone"
	NONE     = "None"

	// Well-known SID Strings
	// https://support.microsoft.com/en-us/help/243330/well-known-security-identifiers-in-windows-operating-systems
	CREATOR_SID_STR  = "S-1-3-0"
	GROUP_SID_STR    = "S-1-3-1"
	EVERYONE_SID_STR = "S-1-1-0"

	// Constants for AceType
	// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-ace_header
	ACCESS_ALLOWED_ACE_TYPE = 0
	ACCESS_DENIED_ACE_TYPE  = 1
)

var (
	advapi32                  = windows.MustLoadDLL("advapi32.dll")
	procSetEntriesInAclW      = advapi32.MustFindProc("SetEntriesInAclW")
	procGetAce                = advapi32.MustFindProc("GetAce")
	procGetNamedSecurityInfoW = advapi32.MustFindProc("GetNamedSecurityInfoW")
)

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-acl
type ACL struct {
	aclRevision byte
	sbz1        byte
	aclSize     uint16
	aceCount    uint16
	sbz2        uint16
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-access_allowed_ace
type accessAllowedAce struct {
	aceType  byte
	aceFlags byte
	aceSize  uint16
	mask     windows.ACCESS_MASK
	sid      windows.SID
}

// Change the permissions of the specified file. Only the nine
// least-significant bytes are used, allowing access by the file's owner, the
// file's group, and everyone else to be explicitly controlled.
func Chmod(name string, fileMode os.FileMode) error {
	creatorOwnerSID, err := windows.StringToSid(CREATOR_SID_STR)
	if err != nil {
		return err
	}
	creatorGroupSID, err := windows.StringToSid(GROUP_SID_STR)
	if err != nil {
		return err
	}
	everyoneSID, err := windows.StringToSid(EVERYONE_SID_STR)
	if err != nil {
		return err
	}

	mode := windows.ACCESS_MASK(fileMode)
	return apply(
		name,
		true,
		false,
		grantSid(((mode&0700)<<23)|((mode&0200)<<9), creatorOwnerSID),
		grantSid(((mode&0070)<<26)|((mode&0020)<<12), creatorGroupSID),
		grantSid(((mode&0007)<<29)|((mode&0002)<<15), everyoneSID),
	)
}

// apply the provided access control entries to a file. If the replace
// parameter is true, existing entries will be overwritten. If the inherit
// parameter is true, the file will inherit ACEs from its parent.
func apply(name string, replace, inherit bool, entries ...windows.EXPLICIT_ACCESS) error {
	var oldAcl windows.Handle
	if !replace {
		var secDesc windows.Handle
		getNamedSecurityInfo(
			name,
			windows.SE_FILE_OBJECT,
			windows.DACL_SECURITY_INFORMATION,
			nil,
			nil,
			&oldAcl,
			nil,
			&secDesc,
		)
		defer windows.LocalFree(secDesc)
	}
	var acl *windows.ACL
	if err := setEntriesInAcl(
		entries,
		oldAcl,
		&acl,
	); err != nil {
		return err
	}
	defer windows.LocalFree((windows.Handle)(unsafe.Pointer(acl)))
	var secInfo windows.SECURITY_INFORMATION
	if !inherit {
		secInfo = windows.PROTECTED_DACL_SECURITY_INFORMATION
	} else {
		secInfo = windows.UNPROTECTED_DACL_SECURITY_INFORMATION
	}
	return windows.SetNamedSecurityInfo(
		name,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|secInfo,
		nil,
		nil,
		acl,
		nil,
	)
}

// GetFileMode returns the mode of the given file.
func GetFileMode(file string) (os.FileMode, error) {
	var (
		daclHandle, secDesc windows.Handle
		owner, group        *windows.SID
	)
	err := getNamedSecurityInfo(
		file,
		windows.SE_FILE_OBJECT,
		windows.OWNER_SECURITY_INFORMATION|windows.GROUP_SECURITY_INFORMATION|windows.DACL_SECURITY_INFORMATION|windows.UNPROTECTED_DACL_SECURITY_INFORMATION,
		&owner,
		&group,
		&daclHandle,
		nil,
		&secDesc,
	)
	if err != nil {
		return 0, err
	}

	defer windows.LocalFree(secDesc)

	dacl := (*ACL)(unsafe.Pointer(daclHandle))
	aces, err := getACEs(dacl)
	if err != nil {
		return 0, err
	}

	allowMode := 0
	denyMode := 0
	for _, ace := range aces {
		accountName, _, _, err := ace.sid.LookupAccount("")
		if err != nil {
			return 0, err
		}

		// LookupAccount may return an empty string.
		if accountName == "" {
			accountName = NONE
		}

		perms := 0
		if (ace.mask & READ_PERMISSIONS) == READ_PERMISSIONS {
			perms = 0x4
		}
		if (ace.mask & WRITE_PERMISSIONS) == WRITE_PERMISSIONS {
			perms |= 0x2
		}
		if (ace.mask & EXECUTE_PERMISSIONS) == EXECUTE_PERMISSIONS {
			perms |= 0x1
		}

		mode := 0
		if owner.Equals(&ace.sid) {
			mode = perms << 6
		}
		if group.Equals(&ace.sid) {
			mode |= perms << 3
		}
		if accountName == EVERYONE {
			mode |= perms
		}

		if ace.aceType == ACCESS_ALLOWED_ACE_TYPE {
			allowMode |= mode
		} else if ace.aceType == ACCESS_DENIED_ACE_TYPE {
			denyMode |= mode
		}
	}

	// Exclude the denied permissions.
	return os.FileMode(allowMode & ^denyMode), nil
}

// https://docs.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setentriesinaclw
func setEntriesInAcl(entries []windows.EXPLICIT_ACCESS, oldAcl windows.Handle, newAcl **windows.ACL) error {
	ret, _, _ := procSetEntriesInAclW.Call(
		uintptr(len(entries)),
		uintptr(unsafe.Pointer(&entries[0])),
		uintptr(oldAcl),
		uintptr(unsafe.Pointer(newAcl)),
	)
	if ret != 0 {
		return windows.Errno(ret)
	}
	return nil
}

// https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-getace
func getACEs(acl *ACL) ([]*accessAllowedAce, error) {
	aces := make([]*accessAllowedAce, acl.aceCount)
	var ace *accessAllowedAce

	for i := uint16(0); i < acl.aceCount; i++ {
		ret, _, _ := procGetAce.Call(
			uintptr(unsafe.Pointer(acl)),
			uintptr(i),
			uintptr(unsafe.Pointer(&ace)),
		)
		if ret == 0 {
			return []*accessAllowedAce{}, windows.GetLastError()
		}

		aceBytes := make([]byte, ace.aceSize)
		copy(aceBytes, (*[(1 << 31) - 1]byte)(unsafe.Pointer(ace))[:len(aceBytes)])
		aces[i] = (*accessAllowedAce)(unsafe.Pointer(&aceBytes[0]))
	}

	return aces, nil
}

// https://docs.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getnamedsecurityinfow
func getNamedSecurityInfo(objectName string, objectType windows.SE_OBJECT_TYPE, secInfo windows.SECURITY_INFORMATION, owner, group **windows.SID, dacl, sacl, secDesc *windows.Handle) error {
	ret, _, _ := procGetNamedSecurityInfoW.Call(
		uintptr(unsafe.Pointer(windows.StringToUTF16Ptr(objectName))),
		uintptr(objectType),
		uintptr(secInfo),
		uintptr(unsafe.Pointer(owner)),
		uintptr(unsafe.Pointer(group)),
		uintptr(unsafe.Pointer(dacl)),
		uintptr(unsafe.Pointer(sacl)),
		uintptr(unsafe.Pointer(secDesc)),
	)
	if ret != 0 {
		return windows.Errno(ret)
	}
	return nil
}

// Create an EXPLICIT_ACCESS instance granting permissions to the provided SID.
func grantSid(accessPermissions windows.ACCESS_MASK, sid *windows.SID) windows.EXPLICIT_ACCESS {
	return accessSid(accessPermissions, windows.GRANT_ACCESS, sid)
}

// Create an EXPLICIT_ACCESS instance denying permissions to the provided SID.
func denySid(accessPermissions windows.ACCESS_MASK, sid *windows.SID) windows.EXPLICIT_ACCESS {
	return accessSid(accessPermissions, windows.DENY_ACCESS, sid)
}

func accessSid(accessPermissions windows.ACCESS_MASK, accessMode windows.ACCESS_MODE, sid *windows.SID) windows.EXPLICIT_ACCESS {
	return windows.EXPLICIT_ACCESS{
		AccessPermissions: accessPermissions,
		AccessMode:        accessMode,
		Inheritance:       windows.SUB_CONTAINERS_AND_OBJECTS_INHERIT,
		Trustee: windows.TRUSTEE{
			TrusteeForm:  windows.TRUSTEE_IS_SID,
			TrusteeValue: windows.TrusteeValueFromSID(sid),
		},
	}
}
