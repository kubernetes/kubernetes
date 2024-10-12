//go:build windows

package winapi

import (
	"errors"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Uint16BufferToSlice wraps a uint16 pointer-and-length into a slice
// for easier interop with Go APIs
func Uint16BufferToSlice(buffer *uint16, bufferLength int) (result []uint16) {
	result = unsafe.Slice(buffer, bufferLength)
	return
}

// UnicodeString corresponds to UNICODE_STRING win32 struct defined here
// https://docs.microsoft.com/en-us/windows/win32/api/ntdef/ns-ntdef-_unicode_string
type UnicodeString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        *uint16
}

// NTSTRSAFE_UNICODE_STRING_MAX_CCH is a constant defined in ntstrsafe.h. This value
// denotes the maximum number of wide chars a path can have.
const NTSTRSAFE_UNICODE_STRING_MAX_CCH = 32767

// String converts a UnicodeString to a golang string
func (uni UnicodeString) String() string {
	// UnicodeString is not guaranteed to be null terminated, therefore
	// use the UnicodeString's Length field
	return windows.UTF16ToString(Uint16BufferToSlice(uni.Buffer, int(uni.Length/2)))
}

// NewUnicodeString allocates a new UnicodeString and copies `s` into
// the buffer of the new UnicodeString.
func NewUnicodeString(s string) (*UnicodeString, error) {
	buf, err := windows.UTF16FromString(s)
	if err != nil {
		return nil, err
	}

	if len(buf) > NTSTRSAFE_UNICODE_STRING_MAX_CCH {
		return nil, syscall.ENAMETOOLONG
	}

	uni := &UnicodeString{
		// The length is in bytes and should not include the trailing null character.
		Length:        uint16((len(buf) - 1) * 2),
		MaximumLength: uint16((len(buf) - 1) * 2),
		Buffer:        &buf[0],
	}
	return uni, nil
}

// ConvertStringSetToSlice is a helper function used to convert the contents of
// `buf` into a string slice. `buf` contains a set of null terminated strings
// with an additional null at the end to indicate the end of the set.
func ConvertStringSetToSlice(buf []byte) ([]string, error) {
	var results []string
	prev := 0
	for i := range buf {
		if buf[i] == 0 {
			if prev == i {
				// found two null characters in a row, return result
				return results, nil
			}
			results = append(results, string(buf[prev:i]))
			prev = i + 1
		}
	}
	return nil, errors.New("string set malformed: missing null terminator at end of buffer")
}

// ParseUtf16LE parses a UTF-16LE byte array into a string (without passing
// through a uint16 or rune array).
func ParseUtf16LE(b []byte) string {
	return windows.UTF16PtrToString((*uint16)(unsafe.Pointer(&b[0])))
}
