//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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

	// Well-known SID Strings
	// https://support.microsoft.com/en-us/help/243330/well-known-security-identifiers-in-windows-operating-systems
	CREATOR_SID_STR  = "S-1-3-2"
	GROUP_SID_STR    = "S-1-3-3"
	EVERYONE_SID_STR = "S-1-1-0"
)

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
		grantSid(((mode&0700)<<23)|((mode&0200)<<9), creatorOwnerSID),
		grantSid(((mode&0070)<<26)|((mode&0020)<<12), creatorGroupSID),
		grantSid(((mode&0007)<<29)|((mode&0002)<<15), everyoneSID),
	)
}

// apply the provided access control entries to a file.
func apply(name string, entries ...windows.EXPLICIT_ACCESS) error {
	acl, err := windows.ACLFromEntries(entries, nil)
	if err != nil {
		return err
	}

	return windows.SetNamedSecurityInfo(
		name,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION,
		nil,
		nil,
		acl,
		nil,
	)
}

// GetFileMode returns the mode of the given file.
func GetFileMode(file string) (os.FileMode, error) {
	descriptor, err := windows.GetNamedSecurityInfo(
		file,
		windows.SE_FILE_OBJECT,
		windows.OWNER_SECURITY_INFORMATION|windows.GROUP_SECURITY_INFORMATION|windows.DACL_SECURITY_INFORMATION|windows.UNPROTECTED_DACL_SECURITY_INFORMATION,
	)
	if err != nil {
		return 0, err
	}

	dacl, _, err := descriptor.DACL()
	if err != nil {
		return 0, err
	}

	owner, _, err := descriptor.Owner()
	if err != nil {
		return 0, err
	}

	group, _, err := descriptor.Group()
	if err != nil {
		return 0, err
	}

	everyone, err := windows.StringToSid(EVERYONE_SID_STR)
	if err != nil {
		return 0, err
	}

	aces, err := windows.GetEntriesFromACL(dacl)
	if err != nil {
		return 0, err
	}

	allowMode := 0
	denyMode := 0
	for _, ace := range aces {
		perms := 0
		if (ace.Mask & READ_PERMISSIONS) == READ_PERMISSIONS {
			perms = 0x4
		}
		if (ace.Mask & WRITE_PERMISSIONS) == WRITE_PERMISSIONS {
			perms |= 0x2
		}
		if (ace.Mask & EXECUTE_PERMISSIONS) == EXECUTE_PERMISSIONS {
			perms |= 0x1
		}

		mode := 0
		if owner.Equals(&ace.Sid) {
			mode = perms << 6
		}
		if group.Equals(&ace.Sid) {
			mode |= perms << 3
		}
		if everyone.Equals(&ace.Sid) {
			mode |= perms
		}

		if ace.Header.AceType == windows.ACCESS_ALLOWED_ACE_TYPE {
			allowMode |= mode
		} else if ace.Header.AceType == windows.ACCESS_DENIED_ACE_TYPE {
			denyMode |= mode
		}
	}

	// Exclude the denied permissions.
	return os.FileMode(allowMode & ^denyMode), nil
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
