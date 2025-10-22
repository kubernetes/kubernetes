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

// Win32_PerfFormattedData_PerfDisk_PhysicalDisk struct
type Win32_PerfFormattedData_PerfDisk_PhysicalDisk struct {
	*Win32_PerfFormattedData

	//
	AvgDiskBytesPerRead uint64

	//
	AvgDiskBytesPerTransfer uint64

	//
	AvgDiskBytesPerWrite uint64

	//
	AvgDiskQueueLength uint64

	//
	AvgDiskReadQueueLength uint64

	//
	AvgDisksecPerRead uint32

	//
	AvgDisksecPerTransfer uint32

	//
	AvgDisksecPerWrite uint32

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
	PercentDiskReadTime uint64

	//
	PercentDiskTime uint64

	//
	PercentDiskWriteTime uint64

	//
	PercentIdleTime uint64

	//
	SplitIOPerSec uint32
}

func NewWin32_PerfFormattedData_PerfDisk_PhysicalDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfDisk_PhysicalDisk{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfDisk_PhysicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfDisk_PhysicalDisk{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAvgDiskBytesPerRead sets the value of AvgDiskBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerRead", (value))
}

// GetAvgDiskBytesPerRead gets the value of AvgDiskBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskBytesPerRead() (value uint64, err error) {
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

// SetAvgDiskBytesPerTransfer sets the value of AvgDiskBytesPerTransfer for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskBytesPerTransfer(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerTransfer", (value))
}

// GetAvgDiskBytesPerTransfer gets the value of AvgDiskBytesPerTransfer for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskBytesPerTransfer() (value uint64, err error) {
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

// SetAvgDiskBytesPerWrite sets the value of AvgDiskBytesPerWrite for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgDiskBytesPerWrite", (value))
}

// GetAvgDiskBytesPerWrite gets the value of AvgDiskBytesPerWrite for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskBytesPerWrite() (value uint64, err error) {
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

// SetAvgDiskQueueLength sets the value of AvgDiskQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskQueueLength", (value))
}

// GetAvgDiskQueueLength gets the value of AvgDiskQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskQueueLength() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskReadQueueLength", (value))
}

// GetAvgDiskReadQueueLength gets the value of AvgDiskReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskReadQueueLength() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDisksecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerRead", (value))
}

// GetAvgDisksecPerRead gets the value of AvgDisksecPerRead for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDisksecPerRead() (value uint32, err error) {
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

// SetAvgDisksecPerTransfer sets the value of AvgDisksecPerTransfer for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDisksecPerTransfer(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerTransfer", (value))
}

// GetAvgDisksecPerTransfer gets the value of AvgDisksecPerTransfer for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDisksecPerTransfer() (value uint32, err error) {
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

// SetAvgDisksecPerWrite sets the value of AvgDisksecPerWrite for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDisksecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgDisksecPerWrite", (value))
}

// GetAvgDisksecPerWrite gets the value of AvgDisksecPerWrite for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDisksecPerWrite() (value uint32, err error) {
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

// SetAvgDiskWriteQueueLength sets the value of AvgDiskWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyAvgDiskWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDiskWriteQueueLength", (value))
}

// GetAvgDiskWriteQueueLength gets the value of AvgDiskWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyAvgDiskWriteQueueLength() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyCurrentDiskQueueLength(value uint32) (err error) {
	return instance.SetProperty("CurrentDiskQueueLength", (value))
}

// GetCurrentDiskQueueLength gets the value of CurrentDiskQueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyCurrentDiskQueueLength() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskBytesPersec", (value))
}

// GetDiskBytesPersec gets the value of DiskBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskReadBytesPersec", (value))
}

// GetDiskReadBytesPersec gets the value of DiskReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskReadBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskReadsPersec(value uint32) (err error) {
	return instance.SetProperty("DiskReadsPersec", (value))
}

// GetDiskReadsPersec gets the value of DiskReadsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskReadsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskTransfersPersec(value uint32) (err error) {
	return instance.SetProperty("DiskTransfersPersec", (value))
}

// GetDiskTransfersPersec gets the value of DiskTransfersPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskTransfersPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskWriteBytesPersec", (value))
}

// GetDiskWriteBytesPersec gets the value of DiskWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskWriteBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyDiskWritesPersec(value uint32) (err error) {
	return instance.SetProperty("DiskWritesPersec", (value))
}

// GetDiskWritesPersec gets the value of DiskWritesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyDiskWritesPersec() (value uint32, err error) {
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

// SetPercentDiskReadTime sets the value of PercentDiskReadTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyPercentDiskReadTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskReadTime", (value))
}

// GetPercentDiskReadTime gets the value of PercentDiskReadTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyPercentDiskReadTime() (value uint64, err error) {
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

// SetPercentDiskTime sets the value of PercentDiskTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyPercentDiskTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskTime", (value))
}

// GetPercentDiskTime gets the value of PercentDiskTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyPercentDiskTime() (value uint64, err error) {
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

// SetPercentDiskWriteTime sets the value of PercentDiskWriteTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyPercentDiskWriteTime(value uint64) (err error) {
	return instance.SetProperty("PercentDiskWriteTime", (value))
}

// GetPercentDiskWriteTime gets the value of PercentDiskWriteTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyPercentDiskWriteTime() (value uint64, err error) {
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

// SetPercentIdleTime sets the value of PercentIdleTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertyPercentIdleTime(value uint64) (err error) {
	return instance.SetProperty("PercentIdleTime", (value))
}

// GetPercentIdleTime gets the value of PercentIdleTime for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertyPercentIdleTime() (value uint64, err error) {
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

// SetSplitIOPerSec sets the value of SplitIOPerSec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) SetPropertySplitIOPerSec(value uint32) (err error) {
	return instance.SetProperty("SplitIOPerSec", (value))
}

// GetSplitIOPerSec gets the value of SplitIOPerSec for the instance
func (instance *Win32_PerfFormattedData_PerfDisk_PhysicalDisk) GetPropertySplitIOPerSec() (value uint32, err error) {
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
