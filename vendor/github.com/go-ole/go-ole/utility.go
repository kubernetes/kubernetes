package ole

import (
	"unicode/utf16"
	"unsafe"
)

// ClassIDFrom retrieves class ID whether given is program ID or application string.
//
// Helper that provides check against both Class ID from Program ID and Class ID from string. It is
// faster, if you know which you are using, to use the individual functions, but this will check
// against available functions for you.
func ClassIDFrom(programID string) (classID *GUID, err error) {
	classID, err = CLSIDFromProgID(programID)
	if err != nil {
		classID, err = CLSIDFromString(programID)
		if err != nil {
			return
		}
	}
	return
}

// BytePtrToString converts byte pointer to a Go string.
func BytePtrToString(p *byte) string {
	a := (*[10000]uint8)(unsafe.Pointer(p))
	i := 0
	for a[i] != 0 {
		i++
	}
	return string(a[:i])
}

// UTF16PtrToString is alias for LpOleStrToString.
//
// Kept for compatibility reasons.
func UTF16PtrToString(p *uint16) string {
	return LpOleStrToString(p)
}

// LpOleStrToString converts COM Unicode to Go string.
func LpOleStrToString(p *uint16) string {
	if p == nil {
		return ""
	}

	length := lpOleStrLen(p)
	a := make([]uint16, length)

	ptr := unsafe.Pointer(p)

	for i := 0; i < int(length); i++ {
		a[i] = *(*uint16)(ptr)
		ptr = unsafe.Pointer(uintptr(ptr) + 2)
	}

	return string(utf16.Decode(a))
}

// BstrToString converts COM binary string to Go string.
func BstrToString(p *uint16) string {
	if p == nil {
		return ""
	}
	length := SysStringLen((*int16)(unsafe.Pointer(p)))
	a := make([]uint16, length)

	ptr := unsafe.Pointer(p)

	for i := 0; i < int(length); i++ {
		a[i] = *(*uint16)(ptr)
		ptr = unsafe.Pointer(uintptr(ptr) + 2)
	}
	return string(utf16.Decode(a))
}

// lpOleStrLen returns the length of Unicode string.
func lpOleStrLen(p *uint16) (length int64) {
	if p == nil {
		return 0
	}

	ptr := unsafe.Pointer(p)

	for i := 0; ; i++ {
		if 0 == *(*uint16)(ptr) {
			length = int64(i)
			break
		}
		ptr = unsafe.Pointer(uintptr(ptr) + 2)
	}
	return
}

// convertHresultToError converts syscall to error, if call is unsuccessful.
func convertHresultToError(hr uintptr, r2 uintptr, ignore error) (err error) {
	if hr != 0 {
		err = NewError(hr)
	}
	return
}
