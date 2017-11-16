// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

type DISPID int32

const (
	DISPID_BEFORENAVIGATE             DISPID = 100
	DISPID_NAVIGATECOMPLETE           DISPID = 101
	DISPID_STATUSTEXTCHANGE           DISPID = 102
	DISPID_QUIT                       DISPID = 103
	DISPID_DOWNLOADCOMPLETE           DISPID = 104
	DISPID_COMMANDSTATECHANGE         DISPID = 105
	DISPID_DOWNLOADBEGIN              DISPID = 106
	DISPID_NEWWINDOW                  DISPID = 107
	DISPID_PROGRESSCHANGE             DISPID = 108
	DISPID_WINDOWMOVE                 DISPID = 109
	DISPID_WINDOWRESIZE               DISPID = 110
	DISPID_WINDOWACTIVATE             DISPID = 111
	DISPID_PROPERTYCHANGE             DISPID = 112
	DISPID_TITLECHANGE                DISPID = 113
	DISPID_TITLEICONCHANGE            DISPID = 114
	DISPID_FRAMEBEFORENAVIGATE        DISPID = 200
	DISPID_FRAMENAVIGATECOMPLETE      DISPID = 201
	DISPID_FRAMENEWWINDOW             DISPID = 204
	DISPID_BEFORENAVIGATE2            DISPID = 250
	DISPID_NEWWINDOW2                 DISPID = 251
	DISPID_NAVIGATECOMPLETE2          DISPID = 252
	DISPID_ONQUIT                     DISPID = 253
	DISPID_ONVISIBLE                  DISPID = 254
	DISPID_ONTOOLBAR                  DISPID = 255
	DISPID_ONMENUBAR                  DISPID = 256
	DISPID_ONSTATUSBAR                DISPID = 257
	DISPID_ONFULLSCREEN               DISPID = 258
	DISPID_DOCUMENTCOMPLETE           DISPID = 259
	DISPID_ONTHEATERMODE              DISPID = 260
	DISPID_ONADDRESSBAR               DISPID = 261
	DISPID_WINDOWSETRESIZABLE         DISPID = 262
	DISPID_WINDOWCLOSING              DISPID = 263
	DISPID_WINDOWSETLEFT              DISPID = 264
	DISPID_WINDOWSETTOP               DISPID = 265
	DISPID_WINDOWSETWIDTH             DISPID = 266
	DISPID_WINDOWSETHEIGHT            DISPID = 267
	DISPID_CLIENTTOHOSTWINDOW         DISPID = 268
	DISPID_SETSECURELOCKICON          DISPID = 269
	DISPID_FILEDOWNLOAD               DISPID = 270
	DISPID_NAVIGATEERROR              DISPID = 271
	DISPID_PRIVACYIMPACTEDSTATECHANGE DISPID = 272
)

var (
	IID_IDispatch = IID{0x00020400, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
)

const (
	DISP_E_MEMBERNOTFOUND = 0x80020003
)

type VARTYPE uint16

const (
	VT_EMPTY            VARTYPE = 0
	VT_NULL             VARTYPE = 1
	VT_I2               VARTYPE = 2
	VT_I4               VARTYPE = 3
	VT_R4               VARTYPE = 4
	VT_R8               VARTYPE = 5
	VT_CY               VARTYPE = 6
	VT_DATE             VARTYPE = 7
	VT_BSTR             VARTYPE = 8
	VT_DISPATCH         VARTYPE = 9
	VT_ERROR            VARTYPE = 10
	VT_BOOL             VARTYPE = 11
	VT_VARIANT          VARTYPE = 12
	VT_UNKNOWN          VARTYPE = 13
	VT_DECIMAL          VARTYPE = 14
	VT_I1               VARTYPE = 16
	VT_UI1              VARTYPE = 17
	VT_UI2              VARTYPE = 18
	VT_UI4              VARTYPE = 19
	VT_I8               VARTYPE = 20
	VT_UI8              VARTYPE = 21
	VT_INT              VARTYPE = 22
	VT_UINT             VARTYPE = 23
	VT_VOID             VARTYPE = 24
	VT_HRESULT          VARTYPE = 25
	VT_PTR              VARTYPE = 26
	VT_SAFEARRAY        VARTYPE = 27
	VT_CARRAY           VARTYPE = 28
	VT_USERDEFINED      VARTYPE = 29
	VT_LPSTR            VARTYPE = 30
	VT_LPWSTR           VARTYPE = 31
	VT_RECORD           VARTYPE = 36
	VT_INT_PTR          VARTYPE = 37
	VT_UINT_PTR         VARTYPE = 38
	VT_FILETIME         VARTYPE = 64
	VT_BLOB             VARTYPE = 65
	VT_STREAM           VARTYPE = 66
	VT_STORAGE          VARTYPE = 67
	VT_STREAMED_OBJECT  VARTYPE = 68
	VT_STORED_OBJECT    VARTYPE = 69
	VT_BLOB_OBJECT      VARTYPE = 70
	VT_CF               VARTYPE = 71
	VT_CLSID            VARTYPE = 72
	VT_VERSIONED_STREAM VARTYPE = 73
	VT_BSTR_BLOB        VARTYPE = 0xfff
	VT_VECTOR           VARTYPE = 0x1000
	VT_ARRAY            VARTYPE = 0x2000
	VT_BYREF            VARTYPE = 0x4000
	VT_RESERVED         VARTYPE = 0x8000
	VT_ILLEGAL          VARTYPE = 0xffff
	VT_ILLEGALMASKED    VARTYPE = 0xfff
	VT_TYPEMASK         VARTYPE = 0xfff
)

type VARIANT struct {
	Vt       VARTYPE
	reserved [14]byte
}

type VARIANTARG VARIANT

type VARIANT_BOOL int16

//type BSTR *uint16

func StringToBSTR(value string) *uint16 /*BSTR*/ {
	// IMPORTANT: Don't forget to free the BSTR value when no longer needed!
	return SysAllocString(value)
}

func BSTRToString(value *uint16 /*BSTR*/) string {
	// ISSUE: Is this really ok?
	bstrArrPtr := (*[200000000]uint16)(unsafe.Pointer(value))

	bstrSlice := make([]uint16, SysStringLen(value))
	copy(bstrSlice, bstrArrPtr[:])

	return syscall.UTF16ToString(bstrSlice)
}

type VAR_I4 struct {
	vt        VARTYPE
	reserved1 [6]byte
	lVal      int32
	reserved2 [4]byte
}

func IntToVariantI4(value int32) *VAR_I4 {
	return &VAR_I4{vt: VT_I4, lVal: value}
}

func VariantI4ToInt(value *VAR_I4) int32 {
	return value.lVal
}

type VAR_BOOL struct {
	vt        VARTYPE
	reserved1 [6]byte
	boolVal   VARIANT_BOOL
	reserved2 [6]byte
}

func BoolToVariantBool(value bool) *VAR_BOOL {
	return &VAR_BOOL{vt: VT_BOOL, boolVal: VARIANT_BOOL(BoolToBOOL(value))}
}

func VariantBoolToBool(value *VAR_BOOL) bool {
	return value.boolVal != 0
}

func StringToVariantBSTR(value string) *VAR_BSTR {
	// IMPORTANT: Don't forget to free the BSTR value when no longer needed!
	return &VAR_BSTR{vt: VT_BSTR, bstrVal: StringToBSTR(value)}
}

func VariantBSTRToString(value *VAR_BSTR) string {
	return BSTRToString(value.bstrVal)
}

type DISPPARAMS struct {
	Rgvarg            *VARIANTARG
	RgdispidNamedArgs *DISPID
	CArgs             int32
	CNamedArgs        int32
}

var (
	// Library
	liboleaut32 uintptr

	// Functions
	sysAllocString uintptr
	sysFreeString  uintptr
	sysStringLen   uintptr
)

func init() {
	// Library
	liboleaut32 = MustLoadLibrary("oleaut32.dll")

	// Functions
	sysAllocString = MustGetProcAddress(liboleaut32, "SysAllocString")
	sysFreeString = MustGetProcAddress(liboleaut32, "SysFreeString")
	sysStringLen = MustGetProcAddress(liboleaut32, "SysStringLen")
}

func SysAllocString(s string) *uint16 /*BSTR*/ {
	ret, _, _ := syscall.Syscall(sysAllocString, 1,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(s))),
		0,
		0)

	return (*uint16) /*BSTR*/ (unsafe.Pointer(ret))
}

func SysFreeString(bstr *uint16 /*BSTR*/) {
	syscall.Syscall(sysFreeString, 1,
		uintptr(unsafe.Pointer(bstr)),
		0,
		0)
}

func SysStringLen(bstr *uint16 /*BSTR*/) uint32 {
	ret, _, _ := syscall.Syscall(sysStringLen, 1,
		uintptr(unsafe.Pointer(bstr)),
		0,
		0)

	return uint32(ret)
}
