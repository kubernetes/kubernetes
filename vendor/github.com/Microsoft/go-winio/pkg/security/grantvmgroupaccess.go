//go:build windows
// +build windows

package security

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

type (
	accessMask          uint32
	accessMode          uint32
	desiredAccess       uint32
	inheritMode         uint32
	objectType          uint32
	shareMode           uint32
	securityInformation uint32
	trusteeForm         uint32
	trusteeType         uint32

	explicitAccess struct {
		accessPermissions accessMask
		accessMode        accessMode
		inheritance       inheritMode
		trustee           trustee
	}

	trustee struct {
		multipleTrustee          *trustee
		multipleTrusteeOperation int32
		trusteeForm              trusteeForm
		trusteeType              trusteeType
		name                     uintptr
	}
)

const (
	accessMaskDesiredPermission accessMask = 1 << 31 // GENERIC_READ

	accessModeGrant accessMode = 1

	desiredAccessReadControl desiredAccess = 0x20000
	desiredAccessWriteDac    desiredAccess = 0x40000

	//cspell:disable-next-line
	gvmga = "GrantVmGroupAccess:"

	inheritModeNoInheritance                  inheritMode = 0x0
	inheritModeSubContainersAndObjectsInherit inheritMode = 0x3

	objectTypeFileObject objectType = 0x1

	securityInformationDACL securityInformation = 0x4

	shareModeRead  shareMode = 0x1
	shareModeWrite shareMode = 0x2

	sidVMGroup = "S-1-5-83-0"

	trusteeFormIsSID trusteeForm = 0

	trusteeTypeWellKnownGroup trusteeType = 5
)

// GrantVMGroupAccess sets the DACL for a specified file or directory to
// include Grant ACE entries for the VM Group SID. This is a golang re-
// implementation of the same function in vmcompute, just not exported in
// RS5. Which kind of sucks. Sucks a lot :/
//
//revive:disable-next-line:var-naming VM, not Vm
func GrantVmGroupAccess(name string) error {
	// Stat (to determine if `name` is a directory).
	s, err := os.Stat(name)
	if err != nil {
		return fmt.Errorf("%s os.Stat %s: %w", gvmga, name, err)
	}

	// Get a handle to the file/directory. Must defer Close on success.
	fd, err := createFile(name, s.IsDir())
	if err != nil {
		return err // Already wrapped
	}
	defer windows.CloseHandle(fd) //nolint:errcheck

	// Get the current DACL and Security Descriptor. Must defer LocalFree on success.
	ot := objectTypeFileObject
	si := securityInformationDACL
	sd := uintptr(0)
	origDACL := uintptr(0)
	if err := getSecurityInfo(fd, uint32(ot), uint32(si), nil, nil, &origDACL, nil, &sd); err != nil {
		return fmt.Errorf("%s GetSecurityInfo %s: %w", gvmga, name, err)
	}
	defer windows.LocalFree(windows.Handle(sd)) //nolint:errcheck

	// Generate a new DACL which is the current DACL with the required ACEs added.
	// Must defer LocalFree on success.
	newDACL, err := generateDACLWithAcesAdded(name, s.IsDir(), origDACL)
	if err != nil {
		return err // Already wrapped
	}
	defer windows.LocalFree(windows.Handle(newDACL)) //nolint:errcheck

	// And finally use SetSecurityInfo to apply the updated DACL.
	if err := setSecurityInfo(fd, uint32(ot), uint32(si), uintptr(0), uintptr(0), newDACL, uintptr(0)); err != nil {
		return fmt.Errorf("%s SetSecurityInfo %s: %w", gvmga, name, err)
	}

	return nil
}

// createFile is a helper function to call [Nt]CreateFile to get a handle to
// the file or directory.
func createFile(name string, isDir bool) (windows.Handle, error) {
	namep, err := windows.UTF16FromString(name)
	if err != nil {
		return windows.InvalidHandle, fmt.Errorf("could not convernt name to UTF-16: %w", err)
	}
	da := uint32(desiredAccessReadControl | desiredAccessWriteDac)
	sm := uint32(shareModeRead | shareModeWrite)
	fa := uint32(windows.FILE_ATTRIBUTE_NORMAL)
	if isDir {
		fa |= windows.FILE_FLAG_BACKUP_SEMANTICS
	}
	fd, err := windows.CreateFile(&namep[0], da, sm, nil, windows.OPEN_EXISTING, fa, 0)
	if err != nil {
		return windows.InvalidHandle, fmt.Errorf("%s windows.CreateFile %s: %w", gvmga, name, err)
	}
	return fd, nil
}

// generateDACLWithAcesAdded generates a new DACL with the two needed ACEs added.
// The caller is responsible for LocalFree of the returned DACL on success.
func generateDACLWithAcesAdded(name string, isDir bool, origDACL uintptr) (uintptr, error) {
	// Generate pointers to the SIDs based on the string SIDs
	sid, err := windows.StringToSid(sidVMGroup)
	if err != nil {
		return 0, fmt.Errorf("%s windows.StringToSid %s %s: %w", gvmga, name, sidVMGroup, err)
	}

	inheritance := inheritModeNoInheritance
	if isDir {
		inheritance = inheritModeSubContainersAndObjectsInherit
	}

	eaArray := []explicitAccess{
		{
			accessPermissions: accessMaskDesiredPermission,
			accessMode:        accessModeGrant,
			inheritance:       inheritance,
			trustee: trustee{
				trusteeForm: trusteeFormIsSID,
				trusteeType: trusteeTypeWellKnownGroup,
				name:        uintptr(unsafe.Pointer(sid)),
			},
		},
	}

	modifiedDACL := uintptr(0)
	if err := setEntriesInAcl(uintptr(uint32(1)), uintptr(unsafe.Pointer(&eaArray[0])), origDACL, &modifiedDACL); err != nil {
		return 0, fmt.Errorf("%s SetEntriesInAcl %s: %w", gvmga, name, err)
	}

	return modifiedDACL, nil
}
