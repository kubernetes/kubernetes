// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

package cim

import (
	"reflect"
	"syscall"
	"unsafe"

	"github.com/go-ole/go-ole"
	"golang.org/x/sys/windows"
	"strings"
)

// Additional calls
var (
	modole32  = windows.NewLazySystemDLL("ole32.dll")
	moduser32 = windows.NewLazySystemDLL("user32.dll")

	procCoInitializeSecurity = modole32.NewProc("CoInitializeSecurity")
	procPeekMessageW         = moduser32.NewProc("PeekMessageW")
)

type RemoveMessageFlags uint32

const (
	PM_NOREMOVE RemoveMessageFlags = 0
	PM_REMOVE   RemoveMessageFlags = 1
	PM_NOYIELD  RemoveMessageFlags = 2
)

type RpcAuthenticationLevel uint32

const (
	RPC_C_AUTHN_LEVEL_DEFAULT       RpcAuthenticationLevel = 0
	RPC_C_AUTHN_LEVEL_NONE          RpcAuthenticationLevel = 1
	RPC_C_AUTHN_LEVEL_CONNECT       RpcAuthenticationLevel = 2
	RPC_C_AUTHN_LEVEL_CALL          RpcAuthenticationLevel = 3
	RPC_C_AUTHN_LEVEL_PKT           RpcAuthenticationLevel = 4
	RPC_C_AUTHN_LEVEL_PKT_INTEGRITY RpcAuthenticationLevel = 5
	RPC_C_AUTHN_LEVEL_PKT_PRIVACY   RpcAuthenticationLevel = 6
)

type RpcImpersonationLevel uint32

const (
	RPC_C_IMP_LEVEL_DEFAULT     RpcImpersonationLevel = 0
	RPC_C_IMP_LEVEL_ANONYMOUS   RpcImpersonationLevel = 1
	RPC_C_IMP_LEVEL_IDENTIFY    RpcImpersonationLevel = 2
	RPC_C_IMP_LEVEL_IMPERSONATE RpcImpersonationLevel = 3
	RPC_C_IMP_LEVEL_DELEGATE    RpcImpersonationLevel = 4
)

type tagEOLE_AUTHENTICATION_CAPABILITIES uint32

const (
	EOAC_NONE              tagEOLE_AUTHENTICATION_CAPABILITIES = 0
	EOAC_MUTUAL_AUTH       tagEOLE_AUTHENTICATION_CAPABILITIES = 0x1
	EOAC_STATIC_CLOAKING   tagEOLE_AUTHENTICATION_CAPABILITIES = 0x20
	EOAC_DYNAMIC_CLOAKING  tagEOLE_AUTHENTICATION_CAPABILITIES = 0x40
	EOAC_ANY_AUTHORITY     tagEOLE_AUTHENTICATION_CAPABILITIES = 0x80
	EOAC_MAKE_FULLSIC      tagEOLE_AUTHENTICATION_CAPABILITIES = 0x100
	EOAC_DEFAULT           tagEOLE_AUTHENTICATION_CAPABILITIES = 0x800
	EOAC_SECURE_REFS       tagEOLE_AUTHENTICATION_CAPABILITIES = 0x2
	EOAC_ACCESS_CONTROL    tagEOLE_AUTHENTICATION_CAPABILITIES = 0x4
	EOAC_APPID             tagEOLE_AUTHENTICATION_CAPABILITIES = 0x8
	EOAC_DYNAMIC           tagEOLE_AUTHENTICATION_CAPABILITIES = 0x10
	EOAC_REQUIRE_FULLSIC   tagEOLE_AUTHENTICATION_CAPABILITIES = 0x200
	EOAC_AUTO_IMPERSONATE  tagEOLE_AUTHENTICATION_CAPABILITIES = 0x400
	EOAC_NO_CUSTOM_MARSHAL tagEOLE_AUTHENTICATION_CAPABILITIES = 0x2000
	EOAC_DISABLE_AAA       tagEOLE_AUTHENTICATION_CAPABILITIES = 0x1000
)

const (
	RPC_E_TOO_LATE uint32 = 0x80010119
)

func GetVariantValue(rawValue *ole.VARIANT) (interface{}, error) {
	array := rawValue.ToArray()
	if array == nil {
		return rawValue.Value(), nil
	}
	return array.ToValueArray(), nil
}

func GetVariantValues(rawValue *ole.VARIANT) ([]interface{}, error) {

	var values []interface{}
	array := rawValue.ToArray()

	if array == nil {
		// Not an array
		values = append(values, rawValue.Value())
	} else {
		values = array.ToValueArray()
	}

	return values, nil
}

type DispParams struct {
	rgvarg            []ole.VARIANT
	rgdispidNamedArgs []int32
}

func GetDispParamsFromRaw(dispparams *DISPPARAMS) *DispParams {

	// convert the DISPPARAMS to a slice header
	params := DispParams{}

	var slice1 []ole.VARIANT
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&slice1))
	sliceHeader.Data = dispparams.rgvarg
	sliceHeader.Len = int(dispparams.cArgs)
	sliceHeader.Cap = int(dispparams.cArgs)
	params.rgvarg = make([]ole.VARIANT, dispparams.cArgs)
	copy(params.rgvarg, slice1)

	var slice2 []int32
	sliceHeader = (*reflect.SliceHeader)(unsafe.Pointer(&slice2))
	sliceHeader.Data = dispparams.rgdispidNamedArgs
	sliceHeader.Len = int(dispparams.cNamedArgs)
	sliceHeader.Cap = int(dispparams.cNamedArgs)
	params.rgdispidNamedArgs = make([]int32, dispparams.cNamedArgs)
	copy(params.rgdispidNamedArgs, slice2)

	return &params
}

func GetVariantArrayAsWmiInstances(rawValues []ole.VARIANT, session *WmiSession) (WmiInstanceCollection, error) {
	wcol := WmiInstanceCollection{}
	for _, rawValue := range rawValues {
		// skip wrong types and empty objects
		if rawValue.VT != ole.VT_DISPATCH || rawValue.Val == 0 {
			continue
		}

		instance, err := CreateWmiInstance(&rawValue, session)
		if err != nil {
			return nil, err
		}
		wcol = append(wcol, instance)
	}

	return wcol, nil
}

func GetVariantArrayValues(rawValues []ole.VARIANT) ([]interface{}, error) {
	var values []interface{}

	for _, rawValue := range rawValues {
		array := rawValue.ToArray()

		if array == nil {
			// Not an array
			values = append(values, rawValue.Value())
		} else {
			values = append(values, array.ToValueArray())
		}
	}

	return values, nil
}

func EscapeQueryValue(rawString string) string {
	// Double the backslash character as per required by the "WHERE" WMI clause
	// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/where-clause
	// Interestingly, double quotes don't seem to need escaping.
	return strings.ReplaceAll(strings.ReplaceAll(rawString, "\\", "\\\\"), "'", "\\'")
}

func FindStringInSlice(stringList []string, value string) (int, bool) {
	for i, item := range stringList {
		if item == value {
			return i, true
		}
	}
	return -1, false
}

// PeekMessage in message queue from runtime.
//
// This function appears to block. PeekMessage does not block.
func PeekMessage(msg *ole.Msg, hwnd uint32, MsgFilterMin uint32, MsgFilterMax uint32, RemoveMsg RemoveMessageFlags) (ret bool, err error) {
	r0, _, err := syscall.Syscall6(procPeekMessageW.Addr(), 5, uintptr(unsafe.Pointer(msg)), uintptr(hwnd), uintptr(MsgFilterMin), uintptr(MsgFilterMax), uintptr(RemoveMsg), 0)
	ret = bool(r0 > 0)
	return
}

func CoInitializeSecurity(authLevel RpcAuthenticationLevel, impLevel RpcImpersonationLevel) (err error) {
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/setting-the-default-process-security-level-using-c-
	hr, _, _ := syscall.Syscall9(procCoInitializeSecurity.Addr(), 9, uintptr(0), ^uintptr(0), uintptr(0), uintptr(0), uintptr(authLevel), uintptr(impLevel), uintptr(0), uintptr(EOAC_NONE), uintptr(0))
	if hr != 0 {
		err = ole.NewError(hr)
	}
	return
}
