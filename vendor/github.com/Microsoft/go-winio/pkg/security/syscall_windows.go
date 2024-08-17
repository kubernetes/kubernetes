package security

//go:generate go run github.com/Microsoft/go-winio/tools/mkwinsyscall -output zsyscall_windows.go syscall_windows.go

//sys getSecurityInfo(handle windows.Handle, objectType uint32, si uint32, ppsidOwner **uintptr, ppsidGroup **uintptr, ppDacl *uintptr, ppSacl *uintptr, ppSecurityDescriptor *uintptr) (win32err error) = advapi32.GetSecurityInfo
//sys setSecurityInfo(handle windows.Handle, objectType uint32, si uint32, psidOwner uintptr, psidGroup uintptr, pDacl uintptr, pSacl uintptr) (win32err error) = advapi32.SetSecurityInfo
//sys setEntriesInAcl(count uintptr, pListOfEEs uintptr, oldAcl uintptr, newAcl *uintptr) (win32err error) = advapi32.SetEntriesInAclW
