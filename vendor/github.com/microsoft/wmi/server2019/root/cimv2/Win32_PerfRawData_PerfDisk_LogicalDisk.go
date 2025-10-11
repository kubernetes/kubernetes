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

// Win32_PerfRawData_PerfDisk_LogicalDisk struct
type Win32_PerfRawData_PerfDisk_LogicalDisk struct {
	*Win32_PerfRawData

	//
	AvgDiskBytesPerRead uint64

	//
	AvgDiskBytesPerRead_Base uint32

	//
	AvgDiskBytesPerTransfer uint64

	//
	AvgDiskBytesPerTransfer_Base uint32

	//
	AvgDiskBytesPerWrite uint64

	//
	AvgDiskBytesPerWrite_Base uint32

	//
	AvgDiskQueueLength uint64

	//
	AvgDiskReadQueueLength uint64

	//
	AvgDisksecPerRead uint32

	//
	AvgDisksecPerRead_Base uint32

	//
	AvgDisksecPerTransfer uint32

	//
	AvgDisksecPerTransfer_Base uint32

	//
	AvgDisksecPerWrite uint32

	//
	AvgDisksecPerWrite_Base uint32

	//
	AvgDiskWriteQueueLength uint64

	//
	CurrentDiskQueueLength uint32

	//
	DiskBytesPersec uint64

	//
	DiskReadBytesPersec uint64

	//
	DiskReadsPersec uint32

	//
	DiskTransfersPersec uint32

	//
	DiskWriteBytesPersec uint64

	//
	DiskWritesPersec uint32

	//
	FreeMegabytes uint32

	//
	PercentDiskReadTime uint64

	//
	PercentDiskReadTime_Base uint64

	//
	PercentDiskTime uint64

	//
	PercentDiskTime_Base uint64

	//
	PercentDiskWriteTime uint64

	//
	PercentDiskWriteTime_Base uint64

	//
	PercentFreeSpace uint32

	//
	PercentFreeSpace_Base uint32

	//
	PercentIdleTime uint64

	//
	PercentIdleTime_Base uint64

	//
	SplitIOPerSec uint32
}

func NewWin32_PerfRawData_PerfDisk_LogicalDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfDisk_LogicalDisk, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfDisk_LogicalDisk{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfDisk_LogicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfDisk_LogicalDisk, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfDisk_LogicalDisk{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvgDiskBytesPerRead sets the value of AvgDiskBytesPerRead for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerRead", (value))
}

// GetAvgDiskBytesPerRead gets the value of AvgDiskBytesPerRead for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerRead")
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

// SetAvgDiskBytesPerRead_Base sets the value of AvgDiskBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDiskBytesPerRead_Base", (value))
}

// GetAvgDiskBytesPerRead_Base gets the value of AvgDiskBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerRead_Base")
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

// SetAvgDiskBytesPerTransfer sets the value of AvgDiskBytesPerTransfer for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerTransfer(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerTransfer", (value))
}

// GetAvgDiskBytesPerTransfer gets the value of AvgDiskBytesPerTransfer for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerTransfer() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerTransfer")
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

// SetAvgDiskBytesPerTransfer_Base sets the value of AvgDiskBytesPerTransfer_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerTransfer_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDiskBytesPerTransfer_Base", (value))
}

// GetAvgDiskBytesPerTransfer_Base gets the value of AvgDiskBytesPerTransfer_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerTransfer_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerTransfer_Base")
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

// SetAvgDiskBytesPerWrite sets the value of AvgDiskBytesPerWrite for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerWrite", (value))
}

// GetAvgDiskBytesPerWrite gets the value of AvgDiskBytesPerWrite for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerWrite() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerWrite")
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

// SetAvgDiskBytesPerWrite_Base sets the value of AvgDiskBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskBytesPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDiskBytesPerWrite_Base", (value))
}

// GetAvgDiskBytesPerWrite_Base gets the value of AvgDiskBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskBytesPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDiskBytesPerWrite_Base")
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

// SetAvgDiskQueueLength sets the value of AvgDiskQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskQueueLength", (value))
}

// GetAvgDiskQueueLength gets the value of AvgDiskQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskQueueLength")
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

// SetAvgDiskReadQueueLength sets the value of AvgDiskReadQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskReadQueueLength", (value))
}

// GetAvgDiskReadQueueLength gets the value of AvgDiskReadQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskReadQueueLength")
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

// SetAvgDisksecPerRead sets the value of AvgDisksecPerRead for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerRead", (value))
}

// GetAvgDisksecPerRead gets the value of AvgDisksecPerRead for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerRead() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerRead")
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

// SetAvgDisksecPerRead_Base sets the value of AvgDisksecPerRead_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerRead_Base", (value))
}

// GetAvgDisksecPerRead_Base gets the value of AvgDisksecPerRead_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerRead_Base")
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

// SetAvgDisksecPerTransfer sets the value of AvgDisksecPerTransfer for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerTransfer(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerTransfer", (value))
}

// GetAvgDisksecPerTransfer gets the value of AvgDisksecPerTransfer for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerTransfer() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerTransfer")
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

// SetAvgDisksecPerTransfer_Base sets the value of AvgDisksecPerTransfer_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerTransfer_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerTransfer_Base", (value))
}

// GetAvgDisksecPerTransfer_Base gets the value of AvgDisksecPerTransfer_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerTransfer_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerTransfer_Base")
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

// SetAvgDisksecPerWrite sets the value of AvgDisksecPerWrite for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerWrite", (value))
}

// GetAvgDisksecPerWrite gets the value of AvgDisksecPerWrite for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerWrite() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerWrite")
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

// SetAvgDisksecPerWrite_Base sets the value of AvgDisksecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDisksecPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerWrite_Base", (value))
}

// GetAvgDisksecPerWrite_Base gets the value of AvgDisksecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDisksecPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDisksecPerWrite_Base")
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

// SetAvgDiskWriteQueueLength sets the value of AvgDiskWriteQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyAvgDiskWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskWriteQueueLength", (value))
}

// GetAvgDiskWriteQueueLength gets the value of AvgDiskWriteQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyAvgDiskWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDiskWriteQueueLength")
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

// SetCurrentDiskQueueLength sets the value of CurrentDiskQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyCurrentDiskQueueLength(value uint32) (err error) {
	return instance.SetProperty("CurrentDiskQueueLength", (value))
}

// GetCurrentDiskQueueLength gets the value of CurrentDiskQueueLength for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyCurrentDiskQueueLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentDiskQueueLength")
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

// SetDiskBytesPersec sets the value of DiskBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskBytesPersec", (value))
}

// GetDiskBytesPersec gets the value of DiskBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskBytesPersec")
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

// SetDiskReadBytesPersec sets the value of DiskReadBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskReadBytesPersec", (value))
}

// GetDiskReadBytesPersec gets the value of DiskReadBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskReadBytesPersec")
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

// SetDiskReadsPersec sets the value of DiskReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskReadsPersec(value uint32) (err error) {
	return instance.SetProperty("DiskReadsPersec", (value))
}

// GetDiskReadsPersec gets the value of DiskReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskReadsPersec")
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

// SetDiskTransfersPersec sets the value of DiskTransfersPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskTransfersPersec(value uint32) (err error) {
	return instance.SetProperty("DiskTransfersPersec", (value))
}

// GetDiskTransfersPersec gets the value of DiskTransfersPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskTransfersPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskTransfersPersec")
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

// SetDiskWriteBytesPersec sets the value of DiskWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskWriteBytesPersec", (value))
}

// GetDiskWriteBytesPersec gets the value of DiskWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskWriteBytesPersec")
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

// SetDiskWritesPersec sets the value of DiskWritesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyDiskWritesPersec(value uint32) (err error) {
	return instance.SetProperty("DiskWritesPersec", (value))
}

// GetDiskWritesPersec gets the value of DiskWritesPersec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyDiskWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskWritesPersec")
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

// SetFreeMegabytes sets the value of FreeMegabytes for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyFreeMegabytes(value uint32) (err error) {
	return instance.SetProperty("FreeMegabytes", (value))
}

// GetFreeMegabytes gets the value of FreeMegabytes for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyFreeMegabytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeMegabytes")
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

// SetPercentDiskReadTime sets the value of PercentDiskReadTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskReadTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskReadTime", (value))
}

// GetPercentDiskReadTime gets the value of PercentDiskReadTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskReadTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskReadTime")
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

// SetPercentDiskReadTime_Base sets the value of PercentDiskReadTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskReadTime_Base(value uint64) (err error) {
	return instance.SetProperty("PercentDiskReadTime_Base", (value))
}

// GetPercentDiskReadTime_Base gets the value of PercentDiskReadTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskReadTime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskReadTime_Base")
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

// SetPercentDiskTime sets the value of PercentDiskTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskTime", (value))
}

// GetPercentDiskTime gets the value of PercentDiskTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskTime")
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

// SetPercentDiskTime_Base sets the value of PercentDiskTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskTime_Base(value uint64) (err error) {
	return instance.SetProperty("PercentDiskTime_Base", (value))
}

// GetPercentDiskTime_Base gets the value of PercentDiskTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskTime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskTime_Base")
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

// SetPercentDiskWriteTime sets the value of PercentDiskWriteTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskWriteTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskWriteTime", (value))
}

// GetPercentDiskWriteTime gets the value of PercentDiskWriteTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskWriteTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskWriteTime")
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

// SetPercentDiskWriteTime_Base sets the value of PercentDiskWriteTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentDiskWriteTime_Base(value uint64) (err error) {
	return instance.SetProperty("PercentDiskWriteTime_Base", (value))
}

// GetPercentDiskWriteTime_Base gets the value of PercentDiskWriteTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentDiskWriteTime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDiskWriteTime_Base")
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

// SetPercentFreeSpace sets the value of PercentFreeSpace for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentFreeSpace(value uint32) (err error) {
	return instance.SetProperty("PercentFreeSpace", (value))
}

// GetPercentFreeSpace gets the value of PercentFreeSpace for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentFreeSpace() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentFreeSpace")
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

// SetPercentFreeSpace_Base sets the value of PercentFreeSpace_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentFreeSpace_Base(value uint32) (err error) {
	return instance.SetProperty("PercentFreeSpace_Base", (value))
}

// GetPercentFreeSpace_Base gets the value of PercentFreeSpace_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentFreeSpace_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentFreeSpace_Base")
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

// SetPercentIdleTime sets the value of PercentIdleTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentIdleTime(value uint64) (err error) {
	return instance.SetProperty("PercentIdleTime", (value))
}

// GetPercentIdleTime gets the value of PercentIdleTime for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentIdleTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentIdleTime")
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

// SetPercentIdleTime_Base sets the value of PercentIdleTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertyPercentIdleTime_Base(value uint64) (err error) {
	return instance.SetProperty("PercentIdleTime_Base", (value))
}

// GetPercentIdleTime_Base gets the value of PercentIdleTime_Base for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertyPercentIdleTime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentIdleTime_Base")
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

// SetSplitIOPerSec sets the value of SplitIOPerSec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) SetPropertySplitIOPerSec(value uint32) (err error) {
	return instance.SetProperty("SplitIOPerSec", (value))
}

// GetSplitIOPerSec gets the value of SplitIOPerSec for the instance
func (instance *Win32_PerfRawData_PerfDisk_LogicalDisk) GetPropertySplitIOPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SplitIOPerSec")
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
