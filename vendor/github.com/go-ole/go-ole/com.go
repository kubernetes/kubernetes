// +build windows

package ole

import (
	"syscall"
	"unicode/utf16"
	"unsafe"
)

var (
	procCoInitialize            = modole32.NewProc("CoInitialize")
	procCoInitializeEx          = modole32.NewProc("CoInitializeEx")
	procCoUninitialize          = modole32.NewProc("CoUninitialize")
	procCoCreateInstance        = modole32.NewProc("CoCreateInstance")
	procCoTaskMemFree           = modole32.NewProc("CoTaskMemFree")
	procCLSIDFromProgID         = modole32.NewProc("CLSIDFromProgID")
	procCLSIDFromString         = modole32.NewProc("CLSIDFromString")
	procStringFromCLSID         = modole32.NewProc("StringFromCLSID")
	procStringFromIID           = modole32.NewProc("StringFromIID")
	procIIDFromString           = modole32.NewProc("IIDFromString")
	procCoGetObject             = modole32.NewProc("CoGetObject")
	procGetUserDefaultLCID      = modkernel32.NewProc("GetUserDefaultLCID")
	procCopyMemory              = modkernel32.NewProc("RtlMoveMemory")
	procVariantInit             = modoleaut32.NewProc("VariantInit")
	procVariantClear            = modoleaut32.NewProc("VariantClear")
	procVariantTimeToSystemTime = modoleaut32.NewProc("VariantTimeToSystemTime")
	procSysAllocString          = modoleaut32.NewProc("SysAllocString")
	procSysAllocStringLen       = modoleaut32.NewProc("SysAllocStringLen")
	procSysFreeString           = modoleaut32.NewProc("SysFreeString")
	procSysStringLen            = modoleaut32.NewProc("SysStringLen")
	procCreateDispTypeInfo      = modoleaut32.NewProc("CreateDispTypeInfo")
	procCreateStdDispatch       = modoleaut32.NewProc("CreateStdDispatch")
	procGetActiveObject         = modoleaut32.NewProc("GetActiveObject")

	procGetMessageW      = moduser32.NewProc("GetMessageW")
	procDispatchMessageW = moduser32.NewProc("DispatchMessageW")
)

// coInitialize initializes COM library on current thread.
//
// MSDN documentation suggests that this function should not be called. Call
// CoInitializeEx() instead. The reason has to do with threading and this
// function is only for single-threaded apartments.
//
// That said, most users of the library have gotten away with just this
// function. If you are experiencing threading issues, then use
// CoInitializeEx().
func coInitialize() (err error) {
	// http://msdn.microsoft.com/en-us/library/windows/desktop/ms678543(v=vs.85).aspx
	// Suggests that no value should be passed to CoInitialized.
	// Could just be Call() since the parameter is optional. <-- Needs testing to be sure.
	hr, _, _ := procCoInitialize.Call(uintptr(0))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// coInitializeEx initializes COM library with concurrency model.
func coInitializeEx(coinit uint32) (err error) {
	// http://msdn.microsoft.com/en-us/library/windows/desktop/ms695279(v=vs.85).aspx
	// Suggests that the first parameter is not only optional but should always be NULL.
	hr, _, _ := procCoInitializeEx.Call(uintptr(0), uintptr(coinit))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// CoInitialize initializes COM library on current thread.
//
// MSDN documentation suggests that this function should not be called. Call
// CoInitializeEx() instead. The reason has to do with threading and this
// function is only for single-threaded apartments.
//
// That said, most users of the library have gotten away with just this
// function. If you are experiencing threading issues, then use
// CoInitializeEx().
func CoInitialize(p uintptr) (err error) {
	// p is ignored and won't be used.
	// Avoid any variable not used errors.
	p = uintptr(0)
	return coInitialize()
}

// CoInitializeEx initializes COM library with concurrency model.
func CoInitializeEx(p uintptr, coinit uint32) (err error) {
	// Avoid any variable not used errors.
	p = uintptr(0)
	return coInitializeEx(coinit)
}

// CoUninitialize uninitializes COM Library.
func CoUninitialize() {
	procCoUninitialize.Call()
}

// CoTaskMemFree frees memory pointer.
func CoTaskMemFree(memptr uintptr) {
	procCoTaskMemFree.Call(memptr)
}

// CLSIDFromProgID retrieves Class Identifier with the given Program Identifier.
//
// The Programmatic Identifier must be registered, because it will be looked up
// in the Windows Registry. The registry entry has the following keys: CLSID,
// Insertable, Protocol and Shell
// (https://msdn.microsoft.com/en-us/library/dd542719(v=vs.85).aspx).
//
// programID identifies the class id with less precision and is not guaranteed
// to be unique. These are usually found in the registry under
// HKEY_LOCAL_MACHINE\SOFTWARE\Classes, usually with the format of
// "Program.Component.Version" with version being optional.
//
// CLSIDFromProgID in Windows API.
func CLSIDFromProgID(progId string) (clsid *GUID, err error) {
	var guid GUID
	lpszProgID := uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(progId)))
	hr, _, _ := procCLSIDFromProgID.Call(lpszProgID, uintptr(unsafe.Pointer(&guid)))
	if hr != 0 {
		err = NewError(hr)
	}
	clsid = &guid
	return
}

// CLSIDFromString retrieves Class ID from string representation.
//
// This is technically the string version of the GUID and will convert the
// string to object.
//
// CLSIDFromString in Windows API.
func CLSIDFromString(str string) (clsid *GUID, err error) {
	var guid GUID
	lpsz := uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(str)))
	hr, _, _ := procCLSIDFromString.Call(lpsz, uintptr(unsafe.Pointer(&guid)))
	if hr != 0 {
		err = NewError(hr)
	}
	clsid = &guid
	return
}

// StringFromCLSID returns GUID formated string from GUID object.
func StringFromCLSID(clsid *GUID) (str string, err error) {
	var p *uint16
	hr, _, _ := procStringFromCLSID.Call(uintptr(unsafe.Pointer(clsid)), uintptr(unsafe.Pointer(&p)))
	if hr != 0 {
		err = NewError(hr)
	}
	str = LpOleStrToString(p)
	return
}

// IIDFromString returns GUID from program ID.
func IIDFromString(progId string) (clsid *GUID, err error) {
	var guid GUID
	lpsz := uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(progId)))
	hr, _, _ := procIIDFromString.Call(lpsz, uintptr(unsafe.Pointer(&guid)))
	if hr != 0 {
		err = NewError(hr)
	}
	clsid = &guid
	return
}

// StringFromIID returns GUID formatted string from GUID object.
func StringFromIID(iid *GUID) (str string, err error) {
	var p *uint16
	hr, _, _ := procStringFromIID.Call(uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&p)))
	if hr != 0 {
		err = NewError(hr)
	}
	str = LpOleStrToString(p)
	return
}

// CreateInstance of single uninitialized object with GUID.
func CreateInstance(clsid *GUID, iid *GUID) (unk *IUnknown, err error) {
	if iid == nil {
		iid = IID_IUnknown
	}
	hr, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(clsid)),
		0,
		CLSCTX_SERVER,
		uintptr(unsafe.Pointer(iid)),
		uintptr(unsafe.Pointer(&unk)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// GetActiveObject retrieves pointer to active object.
func GetActiveObject(clsid *GUID, iid *GUID) (unk *IUnknown, err error) {
	if iid == nil {
		iid = IID_IUnknown
	}
	hr, _, _ := procGetActiveObject.Call(
		uintptr(unsafe.Pointer(clsid)),
		uintptr(unsafe.Pointer(iid)),
		uintptr(unsafe.Pointer(&unk)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

type BindOpts struct {
	CbStruct          uint32
	GrfFlags          uint32
	GrfMode           uint32
	TickCountDeadline uint32
}

// GetObject retrieves pointer to active object.
func GetObject(programID string, bindOpts *BindOpts, iid *GUID) (unk *IUnknown, err error) {
	if bindOpts != nil {
		bindOpts.CbStruct = uint32(unsafe.Sizeof(BindOpts{}))
	}
	if iid == nil {
		iid = IID_IUnknown
	}
	hr, _, _ := procCoGetObject.Call(
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(programID))),
		uintptr(unsafe.Pointer(bindOpts)),
		uintptr(unsafe.Pointer(iid)),
		uintptr(unsafe.Pointer(&unk)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// VariantInit initializes variant.
func VariantInit(v *VARIANT) (err error) {
	hr, _, _ := procVariantInit.Call(uintptr(unsafe.Pointer(v)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// VariantClear clears value in Variant settings to VT_EMPTY.
func VariantClear(v *VARIANT) (err error) {
	hr, _, _ := procVariantClear.Call(uintptr(unsafe.Pointer(v)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// SysAllocString allocates memory for string and copies string into memory.
func SysAllocString(v string) (ss *int16) {
	pss, _, _ := procSysAllocString.Call(uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(v))))
	ss = (*int16)(unsafe.Pointer(pss))
	return
}

// SysAllocStringLen copies up to length of given string returning pointer.
func SysAllocStringLen(v string) (ss *int16) {
	utf16 := utf16.Encode([]rune(v + "\x00"))
	ptr := &utf16[0]

	pss, _, _ := procSysAllocStringLen.Call(uintptr(unsafe.Pointer(ptr)), uintptr(len(utf16)-1))
	ss = (*int16)(unsafe.Pointer(pss))
	return
}

// SysFreeString frees string system memory. This must be called with SysAllocString.
func SysFreeString(v *int16) (err error) {
	hr, _, _ := procSysFreeString.Call(uintptr(unsafe.Pointer(v)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// SysStringLen is the length of the system allocated string.
func SysStringLen(v *int16) uint32 {
	l, _, _ := procSysStringLen.Call(uintptr(unsafe.Pointer(v)))
	return uint32(l)
}

// CreateStdDispatch provides default IDispatch implementation for IUnknown.
//
// This handles default IDispatch implementation for objects. It haves a few
// limitations with only supporting one language. It will also only return
// default exception codes.
func CreateStdDispatch(unk *IUnknown, v uintptr, ptinfo *IUnknown) (disp *IDispatch, err error) {
	hr, _, _ := procCreateStdDispatch.Call(
		uintptr(unsafe.Pointer(unk)),
		v,
		uintptr(unsafe.Pointer(ptinfo)),
		uintptr(unsafe.Pointer(&disp)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// CreateDispTypeInfo provides default ITypeInfo implementation for IDispatch.
//
// This will not handle the full implementation of the interface.
func CreateDispTypeInfo(idata *INTERFACEDATA) (pptinfo *IUnknown, err error) {
	hr, _, _ := procCreateDispTypeInfo.Call(
		uintptr(unsafe.Pointer(idata)),
		uintptr(GetUserDefaultLCID()),
		uintptr(unsafe.Pointer(&pptinfo)))
	if hr != 0 {
		err = NewError(hr)
	}
	return
}

// copyMemory moves location of a block of memory.
func copyMemory(dest unsafe.Pointer, src unsafe.Pointer, length uint32) {
	procCopyMemory.Call(uintptr(dest), uintptr(src), uintptr(length))
}

// GetUserDefaultLCID retrieves current user default locale.
func GetUserDefaultLCID() (lcid uint32) {
	ret, _, _ := procGetUserDefaultLCID.Call()
	lcid = uint32(ret)
	return
}

// GetMessage in message queue from runtime.
//
// This function appears to block. PeekMessage does not block.
func GetMessage(msg *Msg, hwnd uint32, MsgFilterMin uint32, MsgFilterMax uint32) (ret int32, err error) {
	r0, _, err := procGetMessageW.Call(uintptr(unsafe.Pointer(msg)), uintptr(hwnd), uintptr(MsgFilterMin), uintptr(MsgFilterMax))
	ret = int32(r0)
	return
}

// DispatchMessage to window procedure.
func DispatchMessage(msg *Msg) (ret int32) {
	r0, _, _ := procDispatchMessageW.Call(uintptr(unsafe.Pointer(msg)))
	ret = int32(r0)
	return
}
