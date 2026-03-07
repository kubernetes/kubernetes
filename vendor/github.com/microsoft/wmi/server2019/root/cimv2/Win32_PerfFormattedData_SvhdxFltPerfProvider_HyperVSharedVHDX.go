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

// Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX struct
type Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX struct {
	*Win32_PerfFormattedData

	//
	Averagebytesperread uint64

	//
	Averagebytesperrequest uint64

	//
	Averagebytesperwrite uint64

	//
	Averagequeuelength uint64

	//
	Averagereadqueuelength uint64

	//
	AverageSharedVHDXdisklogsize uint64

	//
	AverageSharedVHDXdisktotalsize uint64

	//
	AverageSharedVHDXmounttime uint32

	//
	Averagetimeperread uint32

	//
	Averagetimeperrequest uint32

	//
	Averagetimeperwrite uint32

	//
	Averagewritequeuelength uint64

	//
	Currentqueuelength uint32

	//
	Currentreadqueuelength uint32

	//
	Currentwritequeuelength uint32

	//
	Errorspersecond uint64

	//
	InitiatorHandleOpenspersecond uint32

	//
	ReadBytesPersec uint64

	//
	ReadRequestsPersec uint32

	//
	SharedVHDXMountspersecond uint32

	//
	TotalBytesPersec uint64

	//
	TotalRequestsPersec uint32

	//
	WriteRequestsPersec uint32

	//
	WrittenBytesPersec uint64
}

func NewWin32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDXEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDXEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAveragebytesperread sets the value of Averagebytesperread for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragebytesperread(value uint64) (err error) {
	return instance.SetProperty("Averagebytesperread", (value))
}

// GetAveragebytesperread gets the value of Averagebytesperread for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragebytesperread() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagebytesperread")
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

// SetAveragebytesperrequest sets the value of Averagebytesperrequest for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragebytesperrequest(value uint64) (err error) {
	return instance.SetProperty("Averagebytesperrequest", (value))
}

// GetAveragebytesperrequest gets the value of Averagebytesperrequest for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragebytesperrequest() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagebytesperrequest")
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

// SetAveragebytesperwrite sets the value of Averagebytesperwrite for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragebytesperwrite(value uint64) (err error) {
	return instance.SetProperty("Averagebytesperwrite", (value))
}

// GetAveragebytesperwrite gets the value of Averagebytesperwrite for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragebytesperwrite() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagebytesperwrite")
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

// SetAveragequeuelength sets the value of Averagequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragequeuelength(value uint64) (err error) {
	return instance.SetProperty("Averagequeuelength", (value))
}

// GetAveragequeuelength gets the value of Averagequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragequeuelength() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagequeuelength")
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

// SetAveragereadqueuelength sets the value of Averagereadqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragereadqueuelength(value uint64) (err error) {
	return instance.SetProperty("Averagereadqueuelength", (value))
}

// GetAveragereadqueuelength gets the value of Averagereadqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragereadqueuelength() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagereadqueuelength")
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

// SetAverageSharedVHDXdisklogsize sets the value of AverageSharedVHDXdisklogsize for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAverageSharedVHDXdisklogsize(value uint64) (err error) {
	return instance.SetProperty("AverageSharedVHDXdisklogsize", (value))
}

// GetAverageSharedVHDXdisklogsize gets the value of AverageSharedVHDXdisklogsize for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAverageSharedVHDXdisklogsize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageSharedVHDXdisklogsize")
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

// SetAverageSharedVHDXdisktotalsize sets the value of AverageSharedVHDXdisktotalsize for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAverageSharedVHDXdisktotalsize(value uint64) (err error) {
	return instance.SetProperty("AverageSharedVHDXdisktotalsize", (value))
}

// GetAverageSharedVHDXdisktotalsize gets the value of AverageSharedVHDXdisktotalsize for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAverageSharedVHDXdisktotalsize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageSharedVHDXdisktotalsize")
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

// SetAverageSharedVHDXmounttime sets the value of AverageSharedVHDXmounttime for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAverageSharedVHDXmounttime(value uint32) (err error) {
	return instance.SetProperty("AverageSharedVHDXmounttime", (value))
}

// GetAverageSharedVHDXmounttime gets the value of AverageSharedVHDXmounttime for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAverageSharedVHDXmounttime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageSharedVHDXmounttime")
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

// SetAveragetimeperread sets the value of Averagetimeperread for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragetimeperread(value uint32) (err error) {
	return instance.SetProperty("Averagetimeperread", (value))
}

// GetAveragetimeperread gets the value of Averagetimeperread for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragetimeperread() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagetimeperread")
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

// SetAveragetimeperrequest sets the value of Averagetimeperrequest for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragetimeperrequest(value uint32) (err error) {
	return instance.SetProperty("Averagetimeperrequest", (value))
}

// GetAveragetimeperrequest gets the value of Averagetimeperrequest for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragetimeperrequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagetimeperrequest")
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

// SetAveragetimeperwrite sets the value of Averagetimeperwrite for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragetimeperwrite(value uint32) (err error) {
	return instance.SetProperty("Averagetimeperwrite", (value))
}

// GetAveragetimeperwrite gets the value of Averagetimeperwrite for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragetimeperwrite() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagetimeperwrite")
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

// SetAveragewritequeuelength sets the value of Averagewritequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyAveragewritequeuelength(value uint64) (err error) {
	return instance.SetProperty("Averagewritequeuelength", (value))
}

// GetAveragewritequeuelength gets the value of Averagewritequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyAveragewritequeuelength() (value uint64, err error) {
	retValue, err := instance.GetProperty("Averagewritequeuelength")
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

// SetCurrentqueuelength sets the value of Currentqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyCurrentqueuelength(value uint32) (err error) {
	return instance.SetProperty("Currentqueuelength", (value))
}

// GetCurrentqueuelength gets the value of Currentqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyCurrentqueuelength() (value uint32, err error) {
	retValue, err := instance.GetProperty("Currentqueuelength")
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

// SetCurrentreadqueuelength sets the value of Currentreadqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyCurrentreadqueuelength(value uint32) (err error) {
	return instance.SetProperty("Currentreadqueuelength", (value))
}

// GetCurrentreadqueuelength gets the value of Currentreadqueuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyCurrentreadqueuelength() (value uint32, err error) {
	retValue, err := instance.GetProperty("Currentreadqueuelength")
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

// SetCurrentwritequeuelength sets the value of Currentwritequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyCurrentwritequeuelength(value uint32) (err error) {
	return instance.SetProperty("Currentwritequeuelength", (value))
}

// GetCurrentwritequeuelength gets the value of Currentwritequeuelength for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyCurrentwritequeuelength() (value uint32, err error) {
	retValue, err := instance.GetProperty("Currentwritequeuelength")
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

// SetErrorspersecond sets the value of Errorspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyErrorspersecond(value uint64) (err error) {
	return instance.SetProperty("Errorspersecond", (value))
}

// GetErrorspersecond gets the value of Errorspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyErrorspersecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("Errorspersecond")
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

// SetInitiatorHandleOpenspersecond sets the value of InitiatorHandleOpenspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyInitiatorHandleOpenspersecond(value uint32) (err error) {
	return instance.SetProperty("InitiatorHandleOpenspersecond", (value))
}

// GetInitiatorHandleOpenspersecond gets the value of InitiatorHandleOpenspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyInitiatorHandleOpenspersecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("InitiatorHandleOpenspersecond")
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

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPersec")
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

// SetReadRequestsPersec sets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyReadRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("ReadRequestsPersec", (value))
}

// GetReadRequestsPersec gets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyReadRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadRequestsPersec")
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

// SetSharedVHDXMountspersecond sets the value of SharedVHDXMountspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertySharedVHDXMountspersecond(value uint32) (err error) {
	return instance.SetProperty("SharedVHDXMountspersecond", (value))
}

// GetSharedVHDXMountspersecond gets the value of SharedVHDXMountspersecond for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertySharedVHDXMountspersecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SharedVHDXMountspersecond")
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

// SetTotalBytesPersec sets the value of TotalBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyTotalBytesPersec(value uint64) (err error) {
	return instance.SetProperty("TotalBytesPersec", (value))
}

// GetTotalBytesPersec gets the value of TotalBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyTotalBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalBytesPersec")
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

// SetTotalRequestsPersec sets the value of TotalRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyTotalRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("TotalRequestsPersec", (value))
}

// GetTotalRequestsPersec gets the value of TotalRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyTotalRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalRequestsPersec")
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

// SetWriteRequestsPersec sets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyWriteRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("WriteRequestsPersec", (value))
}

// GetWriteRequestsPersec gets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyWriteRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteRequestsPersec")
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

// SetWrittenBytesPersec sets the value of WrittenBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) SetPropertyWrittenBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WrittenBytesPersec", (value))
}

// GetWrittenBytesPersec gets the value of WrittenBytesPersec for the instance
func (instance *Win32_PerfFormattedData_SvhdxFltPerfProvider_HyperVSharedVHDX) GetPropertyWrittenBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WrittenBytesPersec")
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
