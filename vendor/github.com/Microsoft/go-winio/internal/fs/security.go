package fs

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ne-winnt-security_impersonation_level
type SecurityImpersonationLevel int32 // C default enums underlying type is `int`, which is Go `int32`

// Impersonation levels
const (
	SecurityAnonymous      SecurityImpersonationLevel = 0
	SecurityIdentification SecurityImpersonationLevel = 1
	SecurityImpersonation  SecurityImpersonationLevel = 2
	SecurityDelegation     SecurityImpersonationLevel = 3
)
