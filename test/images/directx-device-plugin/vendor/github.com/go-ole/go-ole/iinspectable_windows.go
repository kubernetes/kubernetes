// +build windows

package ole

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"syscall"
	"unsafe"
)

func (v *IInspectable) GetIids() (iids []*GUID, err error) {
	var count uint32
	var array uintptr
	hr, _, _ := syscall.Syscall(
		v.VTable().GetIIds,
		3,
		uintptr(unsafe.Pointer(v)),
		uintptr(unsafe.Pointer(&count)),
		uintptr(unsafe.Pointer(&array)))
	if hr != 0 {
		err = NewError(hr)
		return
	}
	defer CoTaskMemFree(array)

	iids = make([]*GUID, count)
	byteCount := count * uint32(unsafe.Sizeof(GUID{}))
	slicehdr := reflect.SliceHeader{Data: array, Len: int(byteCount), Cap: int(byteCount)}
	byteSlice := *(*[]byte)(unsafe.Pointer(&slicehdr))
	reader := bytes.NewReader(byteSlice)
	for i := range iids {
		guid := GUID{}
		err = binary.Read(reader, binary.LittleEndian, &guid)
		if err != nil {
			return
		}
		iids[i] = &guid
	}
	return
}

func (v *IInspectable) GetRuntimeClassName() (s string, err error) {
	var hstring HString
	hr, _, _ := syscall.Syscall(
		v.VTable().GetRuntimeClassName,
		2,
		uintptr(unsafe.Pointer(v)),
		uintptr(unsafe.Pointer(&hstring)),
		0)
	if hr != 0 {
		err = NewError(hr)
		return
	}
	s = hstring.String()
	DeleteHString(hstring)
	return
}

func (v *IInspectable) GetTrustLevel() (level uint32, err error) {
	hr, _, _ := syscall.Syscall(
		v.VTable().GetTrustLevel,
		2,
		uintptr(unsafe.Pointer(v)),
		uintptr(unsafe.Pointer(&level)),
		0)
	if hr != 0 {
		err = NewError(hr)
	}
	return
}
