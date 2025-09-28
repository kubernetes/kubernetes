// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_ModuleLoadTrace struct
type Win32_ModuleLoadTrace struct {
	*Win32_ModuleTrace

	//
	DefaultBase uint64

	//
	FileName string

	//
	ImageBase uint64

	//
	ImageChecksum uint32

	//
	ImageSize uint64

	//
	ProcessID uint32

	//
	TimeDateStamp uint32
}

func NewWin32_ModuleLoadTraceEx1(instance *cim.WmiInstance) (newInstance *Win32_ModuleLoadTrace, err error) {
	tmp, err := NewWin32_ModuleTraceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ModuleLoadTrace{
		Win32_ModuleTrace: tmp,
	}
	return
}

func NewWin32_ModuleLoadTraceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ModuleLoadTrace, err error) {
	tmp, err := NewWin32_ModuleTraceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ModuleLoadTrace{
		Win32_ModuleTrace: tmp,
	}
	return
}

// SetDefaultBase sets the value of DefaultBase for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyDefaultBase(value uint64) (err error) {
	return instance.SetProperty("DefaultBase", (value))
}

// GetDefaultBase gets the value of DefaultBase for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyDefaultBase() (value uint64, err error) {
	retValue, err := instance.GetProperty("DefaultBase")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFileName sets the value of FileName for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyFileName(value string) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyFileName() (value string, err error) {
	retValue, err := instance.GetProperty("FileName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetImageBase sets the value of ImageBase for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyImageBase(value uint64) (err error) {
	return instance.SetProperty("ImageBase", (value))
}

// GetImageBase gets the value of ImageBase for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyImageBase() (value uint64, err error) {
	retValue, err := instance.GetProperty("ImageBase")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetImageChecksum sets the value of ImageChecksum for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyImageChecksum(value uint32) (err error) {
	return instance.SetProperty("ImageChecksum", (value))
}

// GetImageChecksum gets the value of ImageChecksum for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyImageChecksum() (value uint32, err error) {
	retValue, err := instance.GetProperty("ImageChecksum")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetImageSize sets the value of ImageSize for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyImageSize(value uint64) (err error) {
	return instance.SetProperty("ImageSize", (value))
}

// GetImageSize gets the value of ImageSize for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyImageSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("ImageSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetProcessID sets the value of ProcessID for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyProcessID(value uint32) (err error) {
	return instance.SetProperty("ProcessID", (value))
}

// GetProcessID gets the value of ProcessID for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyProcessID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTimeDateStamp sets the value of TimeDateStamp for the instance
func (instance *Win32_ModuleLoadTrace) SetPropertyTimeDateStamp(value uint32) (err error) {
	return instance.SetProperty("TimeDateStamp", (value))
}

// GetTimeDateStamp gets the value of TimeDateStamp for the instance
func (instance *Win32_ModuleLoadTrace) GetPropertyTimeDateStamp() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimeDateStamp")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}
