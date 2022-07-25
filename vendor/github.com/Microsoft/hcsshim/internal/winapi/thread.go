package winapi

// HANDLE CreateRemoteThread(
// 	HANDLE                 hProcess,
// 	LPSECURITY_ATTRIBUTES  lpThreadAttributes,
// 	SIZE_T                 dwStackSize,
// 	LPTHREAD_START_ROUTINE lpStartAddress,
// 	LPVOID                 lpParameter,
// 	DWORD                  dwCreationFlags,
// 	LPDWORD                lpThreadId
// );
//sys CreateRemoteThread(process windows.Handle, sa *windows.SecurityAttributes, stackSize uint32, startAddr uintptr, parameter uintptr, creationFlags uint32, threadID *uint32) (handle windows.Handle, err error) = kernel32.CreateRemoteThread
