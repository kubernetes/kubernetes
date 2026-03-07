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

// Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem struct
type Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem struct {
	*Win32_PerfFormattedData

	//
	CreateFile uint64

	//
	CreateFilePersec uint64

	//
	FilesInvalidatedDuringResume uint64

	//
	FilesInvalidatedOther uint64

	//
	FilesOpened uint32

	//
	Flushes uint64

	//
	FlushesPersec uint64

	//
	IOReadAvgQueueLength uint64

	//
	IOReadBytes uint64

	//
	IOReadBytesPersec uint64

	//
	IOReadLatency uint32

	//
	IOReadQueueLength uint64

	//
	IOReads uint64

	//
	IOReadsPersec uint64

	//
	IOSingleReads uint64

	//
	IOSingleReadsPersec uint64

	//
	IOSingleWrites uint64

	//
	IOSingleWritesPersec uint64

	//
	IOSplitReads uint64

	//
	IOSplitReadsPersec uint64

	//
	IOSplitWrites uint64

	//
	IOSplitWritesPersec uint64

	//
	IOWriteAvgQueueLength uint64

	//
	IOWriteBytes uint64

	//
	IOWriteBytesPersec uint64

	//
	IOWriteLatency uint32

	//
	IOWriteQueueLength uint64

	//
	IOWrites uint64

	//
	IOWritesPersec uint64

	//
	MetadataIO uint64

	//
	MetadataIOPersec uint64

	//
	ReadLatency uint32

	//
	ReadQueueLength uint64

	//
	Reads uint64

	//
	ReadsPersec uint64

	//
	RedirectedReadBytes uint64

	//
	RedirectedReadBytesPersec uint64

	//
	RedirectedReadLatency uint32

	//
	RedirectedReadQueueLength uint64

	//
	RedirectedReads uint64

	//
	RedirectedReadsAvgQueueLength uint64

	//
	RedirectedReadsPersec uint64

	//
	RedirectedWriteBytes uint64

	//
	RedirectedWriteBytesPersec uint64

	//
	RedirectedWriteLatency uint32

	//
	RedirectedWriteQueueLength uint64

	//
	RedirectedWrites uint64

	//
	RedirectedWritesAvgQueueLength uint64

	//
	RedirectedWritesPersec uint64

	//
	VolumePauseCountDisk uint64

	//
	VolumePauseCountNetwork uint64

	//
	VolumePauseCountOther uint64

	//
	VolumePauseCountTotal uint64

	//
	VolumeState uint32

	//
	WriteLatency uint32

	//
	WriteQueueLength uint64

	//
	Writes uint64

	//
	WritesPersec uint64
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystemEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCreateFile sets the value of CreateFile for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyCreateFile(value uint64) (err error) {
	return instance.SetProperty("CreateFile", (value))
}

// GetCreateFile gets the value of CreateFile for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyCreateFile() (value uint64, err error) {
	retValue, err := instance.GetProperty("CreateFile")
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

// SetCreateFilePersec sets the value of CreateFilePersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyCreateFilePersec(value uint64) (err error) {
	return instance.SetProperty("CreateFilePersec", (value))
}

// GetCreateFilePersec gets the value of CreateFilePersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyCreateFilePersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CreateFilePersec")
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

// SetFilesInvalidatedDuringResume sets the value of FilesInvalidatedDuringResume for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyFilesInvalidatedDuringResume(value uint64) (err error) {
	return instance.SetProperty("FilesInvalidatedDuringResume", (value))
}

// GetFilesInvalidatedDuringResume gets the value of FilesInvalidatedDuringResume for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyFilesInvalidatedDuringResume() (value uint64, err error) {
	retValue, err := instance.GetProperty("FilesInvalidatedDuringResume")
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

// SetFilesInvalidatedOther sets the value of FilesInvalidatedOther for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyFilesInvalidatedOther(value uint64) (err error) {
	return instance.SetProperty("FilesInvalidatedOther", (value))
}

// GetFilesInvalidatedOther gets the value of FilesInvalidatedOther for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyFilesInvalidatedOther() (value uint64, err error) {
	retValue, err := instance.GetProperty("FilesInvalidatedOther")
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

// SetFilesOpened sets the value of FilesOpened for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyFilesOpened(value uint32) (err error) {
	return instance.SetProperty("FilesOpened", (value))
}

// GetFilesOpened gets the value of FilesOpened for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyFilesOpened() (value uint32, err error) {
	retValue, err := instance.GetProperty("FilesOpened")
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

// SetFlushes sets the value of Flushes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyFlushes(value uint64) (err error) {
	return instance.SetProperty("Flushes", (value))
}

// GetFlushes gets the value of Flushes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyFlushes() (value uint64, err error) {
	retValue, err := instance.GetProperty("Flushes")
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

// SetFlushesPersec sets the value of FlushesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyFlushesPersec(value uint64) (err error) {
	return instance.SetProperty("FlushesPersec", (value))
}

// GetFlushesPersec gets the value of FlushesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyFlushesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlushesPersec")
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

// SetIOReadAvgQueueLength sets the value of IOReadAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("IOReadAvgQueueLength", (value))
}

// GetIOReadAvgQueueLength gets the value of IOReadAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadAvgQueueLength")
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

// SetIOReadBytes sets the value of IOReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadBytes(value uint64) (err error) {
	return instance.SetProperty("IOReadBytes", (value))
}

// GetIOReadBytes gets the value of IOReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytes")
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

// SetIOReadBytesPersec sets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadBytesPersec", (value))
}

// GetIOReadBytesPersec gets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytesPersec")
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

// SetIOReadLatency sets the value of IOReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadLatency(value uint32) (err error) {
	return instance.SetProperty("IOReadLatency", (value))
}

// GetIOReadLatency gets the value of IOReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOReadLatency")
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

// SetIOReadQueueLength sets the value of IOReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("IOReadQueueLength", (value))
}

// GetIOReadQueueLength gets the value of IOReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadQueueLength")
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

// SetIOReads sets the value of IOReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReads(value uint64) (err error) {
	return instance.SetProperty("IOReads", (value))
}

// GetIOReads gets the value of IOReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReads")
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

// SetIOReadsPersec sets the value of IOReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOReadsPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadsPersec", (value))
}

// GetIOReadsPersec gets the value of IOReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadsPersec")
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

// SetIOSingleReads sets the value of IOSingleReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSingleReads(value uint64) (err error) {
	return instance.SetProperty("IOSingleReads", (value))
}

// GetIOSingleReads gets the value of IOSingleReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSingleReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSingleReads")
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

// SetIOSingleReadsPersec sets the value of IOSingleReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSingleReadsPersec(value uint64) (err error) {
	return instance.SetProperty("IOSingleReadsPersec", (value))
}

// GetIOSingleReadsPersec gets the value of IOSingleReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSingleReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSingleReadsPersec")
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

// SetIOSingleWrites sets the value of IOSingleWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSingleWrites(value uint64) (err error) {
	return instance.SetProperty("IOSingleWrites", (value))
}

// GetIOSingleWrites gets the value of IOSingleWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSingleWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSingleWrites")
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

// SetIOSingleWritesPersec sets the value of IOSingleWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSingleWritesPersec(value uint64) (err error) {
	return instance.SetProperty("IOSingleWritesPersec", (value))
}

// GetIOSingleWritesPersec gets the value of IOSingleWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSingleWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSingleWritesPersec")
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

// SetIOSplitReads sets the value of IOSplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSplitReads(value uint64) (err error) {
	return instance.SetProperty("IOSplitReads", (value))
}

// GetIOSplitReads gets the value of IOSplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSplitReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSplitReads")
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

// SetIOSplitReadsPersec sets the value of IOSplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSplitReadsPersec(value uint64) (err error) {
	return instance.SetProperty("IOSplitReadsPersec", (value))
}

// GetIOSplitReadsPersec gets the value of IOSplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSplitReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSplitReadsPersec")
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

// SetIOSplitWrites sets the value of IOSplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSplitWrites(value uint64) (err error) {
	return instance.SetProperty("IOSplitWrites", (value))
}

// GetIOSplitWrites gets the value of IOSplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSplitWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSplitWrites")
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

// SetIOSplitWritesPersec sets the value of IOSplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOSplitWritesPersec(value uint64) (err error) {
	return instance.SetProperty("IOSplitWritesPersec", (value))
}

// GetIOSplitWritesPersec gets the value of IOSplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOSplitWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOSplitWritesPersec")
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

// SetIOWriteAvgQueueLength sets the value of IOWriteAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWriteAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("IOWriteAvgQueueLength", (value))
}

// GetIOWriteAvgQueueLength gets the value of IOWriteAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWriteAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteAvgQueueLength")
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

// SetIOWriteBytes sets the value of IOWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWriteBytes(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytes", (value))
}

// GetIOWriteBytes gets the value of IOWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteBytes")
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

// SetIOWriteBytesPersec sets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytesPersec", (value))
}

// GetIOWriteBytesPersec gets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteBytesPersec")
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

// SetIOWriteLatency sets the value of IOWriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWriteLatency(value uint32) (err error) {
	return instance.SetProperty("IOWriteLatency", (value))
}

// GetIOWriteLatency gets the value of IOWriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOWriteLatency")
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

// SetIOWriteQueueLength sets the value of IOWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("IOWriteQueueLength", (value))
}

// GetIOWriteQueueLength gets the value of IOWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteQueueLength")
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

// SetIOWrites sets the value of IOWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWrites(value uint64) (err error) {
	return instance.SetProperty("IOWrites", (value))
}

// GetIOWrites gets the value of IOWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWrites")
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

// SetIOWritesPersec sets the value of IOWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyIOWritesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWritesPersec", (value))
}

// GetIOWritesPersec gets the value of IOWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyIOWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWritesPersec")
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

// SetMetadataIO sets the value of MetadataIO for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyMetadataIO(value uint64) (err error) {
	return instance.SetProperty("MetadataIO", (value))
}

// GetMetadataIO gets the value of MetadataIO for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyMetadataIO() (value uint64, err error) {
	retValue, err := instance.GetProperty("MetadataIO")
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

// SetMetadataIOPersec sets the value of MetadataIOPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyMetadataIOPersec(value uint64) (err error) {
	return instance.SetProperty("MetadataIOPersec", (value))
}

// GetMetadataIOPersec gets the value of MetadataIOPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyMetadataIOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MetadataIOPersec")
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

// SetReadLatency sets the value of ReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyReadLatency(value uint32) (err error) {
	return instance.SetProperty("ReadLatency", (value))
}

// GetReadLatency gets the value of ReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadLatency")
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

// SetReadQueueLength sets the value of ReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("ReadQueueLength", (value))
}

// GetReadQueueLength gets the value of ReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadQueueLength")
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

// SetReads sets the value of Reads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyReads(value uint64) (err error) {
	return instance.SetProperty("Reads", (value))
}

// GetReads gets the value of Reads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads")
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

// SetReadsPersec sets the value of ReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyReadsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec", (value))
}

// GetReadsPersec gets the value of ReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec")
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

// SetRedirectedReadBytes sets the value of RedirectedReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadBytes(value uint64) (err error) {
	return instance.SetProperty("RedirectedReadBytes", (value))
}

// GetRedirectedReadBytes gets the value of RedirectedReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReadBytes")
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

// SetRedirectedReadBytesPersec sets the value of RedirectedReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RedirectedReadBytesPersec", (value))
}

// GetRedirectedReadBytesPersec gets the value of RedirectedReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReadBytesPersec")
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

// SetRedirectedReadLatency sets the value of RedirectedReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadLatency(value uint32) (err error) {
	return instance.SetProperty("RedirectedReadLatency", (value))
}

// GetRedirectedReadLatency gets the value of RedirectedReadLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("RedirectedReadLatency")
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

// SetRedirectedReadQueueLength sets the value of RedirectedReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("RedirectedReadQueueLength", (value))
}

// GetRedirectedReadQueueLength gets the value of RedirectedReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReadQueueLength")
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

// SetRedirectedReads sets the value of RedirectedReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReads(value uint64) (err error) {
	return instance.SetProperty("RedirectedReads", (value))
}

// GetRedirectedReads gets the value of RedirectedReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReads")
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

// SetRedirectedReadsAvgQueueLength sets the value of RedirectedReadsAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadsAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("RedirectedReadsAvgQueueLength", (value))
}

// GetRedirectedReadsAvgQueueLength gets the value of RedirectedReadsAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadsAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReadsAvgQueueLength")
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

// SetRedirectedReadsPersec sets the value of RedirectedReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedReadsPersec(value uint64) (err error) {
	return instance.SetProperty("RedirectedReadsPersec", (value))
}

// GetRedirectedReadsPersec gets the value of RedirectedReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedReadsPersec")
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

// SetRedirectedWriteBytes sets the value of RedirectedWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWriteBytes(value uint64) (err error) {
	return instance.SetProperty("RedirectedWriteBytes", (value))
}

// GetRedirectedWriteBytes gets the value of RedirectedWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWriteBytes")
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

// SetRedirectedWriteBytesPersec sets the value of RedirectedWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RedirectedWriteBytesPersec", (value))
}

// GetRedirectedWriteBytesPersec gets the value of RedirectedWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWriteBytesPersec")
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

// SetRedirectedWriteLatency sets the value of RedirectedWriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWriteLatency(value uint32) (err error) {
	return instance.SetProperty("RedirectedWriteLatency", (value))
}

// GetRedirectedWriteLatency gets the value of RedirectedWriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("RedirectedWriteLatency")
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

// SetRedirectedWriteQueueLength sets the value of RedirectedWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("RedirectedWriteQueueLength", (value))
}

// GetRedirectedWriteQueueLength gets the value of RedirectedWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWriteQueueLength")
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

// SetRedirectedWrites sets the value of RedirectedWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWrites(value uint64) (err error) {
	return instance.SetProperty("RedirectedWrites", (value))
}

// GetRedirectedWrites gets the value of RedirectedWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWrites")
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

// SetRedirectedWritesAvgQueueLength sets the value of RedirectedWritesAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWritesAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("RedirectedWritesAvgQueueLength", (value))
}

// GetRedirectedWritesAvgQueueLength gets the value of RedirectedWritesAvgQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWritesAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWritesAvgQueueLength")
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

// SetRedirectedWritesPersec sets the value of RedirectedWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyRedirectedWritesPersec(value uint64) (err error) {
	return instance.SetProperty("RedirectedWritesPersec", (value))
}

// GetRedirectedWritesPersec gets the value of RedirectedWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyRedirectedWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RedirectedWritesPersec")
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

// SetVolumePauseCountDisk sets the value of VolumePauseCountDisk for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyVolumePauseCountDisk(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountDisk", (value))
}

// GetVolumePauseCountDisk gets the value of VolumePauseCountDisk for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyVolumePauseCountDisk() (value uint64, err error) {
	retValue, err := instance.GetProperty("VolumePauseCountDisk")
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

// SetVolumePauseCountNetwork sets the value of VolumePauseCountNetwork for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyVolumePauseCountNetwork(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountNetwork", (value))
}

// GetVolumePauseCountNetwork gets the value of VolumePauseCountNetwork for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyVolumePauseCountNetwork() (value uint64, err error) {
	retValue, err := instance.GetProperty("VolumePauseCountNetwork")
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

// SetVolumePauseCountOther sets the value of VolumePauseCountOther for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyVolumePauseCountOther(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountOther", (value))
}

// GetVolumePauseCountOther gets the value of VolumePauseCountOther for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyVolumePauseCountOther() (value uint64, err error) {
	retValue, err := instance.GetProperty("VolumePauseCountOther")
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

// SetVolumePauseCountTotal sets the value of VolumePauseCountTotal for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyVolumePauseCountTotal(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountTotal", (value))
}

// GetVolumePauseCountTotal gets the value of VolumePauseCountTotal for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyVolumePauseCountTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("VolumePauseCountTotal")
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

// SetVolumeState sets the value of VolumeState for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyVolumeState(value uint32) (err error) {
	return instance.SetProperty("VolumeState", (value))
}

// GetVolumeState gets the value of VolumeState for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyVolumeState() (value uint32, err error) {
	retValue, err := instance.GetProperty("VolumeState")
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

// SetWriteLatency sets the value of WriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyWriteLatency(value uint32) (err error) {
	return instance.SetProperty("WriteLatency", (value))
}

// GetWriteLatency gets the value of WriteLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteLatency")
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

// SetWriteQueueLength sets the value of WriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("WriteQueueLength", (value))
}

// GetWriteQueueLength gets the value of WriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteQueueLength")
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

// SetWrites sets the value of Writes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyWrites(value uint64) (err error) {
	return instance.SetProperty("Writes", (value))
}

// GetWrites gets the value of Writes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes")
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

// SetWritesPersec sets the value of WritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) SetPropertyWritesPersec(value uint64) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFileSystem) GetPropertyWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec")
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
