package winapi

const PROCESS_ALL_ACCESS uint32 = 2097151

// DWORD GetProcessImageFileNameW(
//	HANDLE hProcess,
//	LPWSTR lpImageFileName,
//	DWORD  nSize
// );
//sys GetProcessImageFileName(hProcess windows.Handle, imageFileName *uint16, nSize uint32) (size uint32, err error) = kernel32.GetProcessImageFileNameW
