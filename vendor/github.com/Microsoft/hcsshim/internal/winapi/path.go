package winapi

// DWORD SearchPathW(
// 	LPCWSTR lpPath,
// 	LPCWSTR lpFileName,
// 	LPCWSTR lpExtension,
// 	DWORD   nBufferLength,
// 	LPWSTR  lpBuffer,
// 	LPWSTR  *lpFilePart
// );
//sys SearchPath(lpPath *uint16, lpFileName *uint16, lpExtension *uint16, nBufferLength uint32, lpBuffer *uint16, lpFilePath *uint16) (size uint32, err error) = kernel32.SearchPathW
