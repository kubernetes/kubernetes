package winapi

// BOOL LogonUserA(
// 	LPCWSTR  lpszUsername,
// 	LPCWSTR  lpszDomain,
// 	LPCWSTR  lpszPassword,
// 	DWORD   dwLogonType,
// 	DWORD   dwLogonProvider,
// 	PHANDLE phToken
// );
//
//sys LogonUser(username *uint16, domain *uint16, password *uint16, logonType uint32, logonProvider uint32, token *windows.Token) (err error) = advapi32.LogonUserW

// Logon types
const (
	LOGON32_LOGON_INTERACTIVE       uint32 = 2
	LOGON32_LOGON_NETWORK           uint32 = 3
	LOGON32_LOGON_BATCH             uint32 = 4
	LOGON32_LOGON_SERVICE           uint32 = 5
	LOGON32_LOGON_UNLOCK            uint32 = 7
	LOGON32_LOGON_NETWORK_CLEARTEXT uint32 = 8
	LOGON32_LOGON_NEW_CREDENTIALS   uint32 = 9
)

// Logon providers
const (
	LOGON32_PROVIDER_DEFAULT uint32 = 0
	LOGON32_PROVIDER_WINNT40 uint32 = 2
	LOGON32_PROVIDER_WINNT50 uint32 = 3
)
