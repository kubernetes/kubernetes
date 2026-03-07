// +build !windows

package ole

import (
	"time"
	"unsafe"
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
func coInitialize() error {
	return NewError(E_NOTIMPL)
}

// coInitializeEx initializes COM library with concurrency model.
func coInitializeEx(coinit uint32) error {
	return NewError(E_NOTIMPL)
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
func CoInitialize(p uintptr) error {
	return NewError(E_NOTIMPL)
}

// CoInitializeEx initializes COM library with concurrency model.
func CoInitializeEx(p uintptr, coinit uint32) error {
	return NewError(E_NOTIMPL)
}

// CoUninitialize uninitializes COM Library.
func CoUninitialize() {}

// CoTaskMemFree frees memory pointer.
func CoTaskMemFree(memptr uintptr) {}

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
func CLSIDFromProgID(progId string) (*GUID, error) {
	return nil, NewError(E_NOTIMPL)
}

// CLSIDFromString retrieves Class ID from string representation.
//
// This is technically the string version of the GUID and will convert the
// string to object.
//
// CLSIDFromString in Windows API.
func CLSIDFromString(str string) (*GUID, error) {
	return nil, NewError(E_NOTIMPL)
}

// StringFromCLSID returns GUID formated string from GUID object.
func StringFromCLSID(clsid *GUID) (string, error) {
	return "", NewError(E_NOTIMPL)
}

// IIDFromString returns GUID from program ID.
func IIDFromString(progId string) (*GUID, error) {
	return nil, NewError(E_NOTIMPL)
}

// StringFromIID returns GUID formatted string from GUID object.
func StringFromIID(iid *GUID) (string, error) {
	return "", NewError(E_NOTIMPL)
}

// CreateInstance of single uninitialized object with GUID.
func CreateInstance(clsid *GUID, iid *GUID) (*IUnknown, error) {
	return nil, NewError(E_NOTIMPL)
}

// GetActiveObject retrieves pointer to active object.
func GetActiveObject(clsid *GUID, iid *GUID) (*IUnknown, error) {
	return nil, NewError(E_NOTIMPL)
}

// VariantInit initializes variant.
func VariantInit(v *VARIANT) error {
	return NewError(E_NOTIMPL)
}

// VariantClear clears value in Variant settings to VT_EMPTY.
func VariantClear(v *VARIANT) error {
	return NewError(E_NOTIMPL)
}

// SysAllocString allocates memory for string and copies string into memory.
func SysAllocString(v string) *int16 {
	u := int16(0)
	return &u
}

// SysAllocStringLen copies up to length of given string returning pointer.
func SysAllocStringLen(v string) *int16 {
	u := int16(0)
	return &u
}

// SysFreeString frees string system memory. This must be called with SysAllocString.
func SysFreeString(v *int16) error {
	return NewError(E_NOTIMPL)
}

// SysStringLen is the length of the system allocated string.
func SysStringLen(v *int16) uint32 {
	return uint32(0)
}

// CreateStdDispatch provides default IDispatch implementation for IUnknown.
//
// This handles default IDispatch implementation for objects. It haves a few
// limitations with only supporting one language. It will also only return
// default exception codes.
func CreateStdDispatch(unk *IUnknown, v uintptr, ptinfo *IUnknown) (*IDispatch, error) {
	return nil, NewError(E_NOTIMPL)
}

// CreateDispTypeInfo provides default ITypeInfo implementation for IDispatch.
//
// This will not handle the full implementation of the interface.
func CreateDispTypeInfo(idata *INTERFACEDATA) (*IUnknown, error) {
	return nil, NewError(E_NOTIMPL)
}

// copyMemory moves location of a block of memory.
func copyMemory(dest unsafe.Pointer, src unsafe.Pointer, length uint32) {}

// GetUserDefaultLCID retrieves current user default locale.
func GetUserDefaultLCID() uint32 {
	return uint32(0)
}

// GetMessage in message queue from runtime.
//
// This function appears to block. PeekMessage does not block.
func GetMessage(msg *Msg, hwnd uint32, MsgFilterMin uint32, MsgFilterMax uint32) (int32, error) {
	return int32(0), NewError(E_NOTIMPL)
}

// DispatchMessage to window procedure.
func DispatchMessage(msg *Msg) int32 {
	return int32(0)
}

func GetVariantDate(value uint64) (time.Time, error) {
	return time.Now(), NewError(E_NOTIMPL)
}
