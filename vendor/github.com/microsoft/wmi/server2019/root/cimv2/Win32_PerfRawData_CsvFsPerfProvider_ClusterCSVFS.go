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

// Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS struct
type Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS struct {
	*Win32_PerfRawData

	//
	AvgBytesPerRead uint64

	//
	AvgBytesPerRead_Base uint32

	//
	AvgBytesPerWrite uint64

	//
	AvgBytesPerWrite_Base uint32

	//
	AvgReadQueueLength uint64

	//
	AvgsecPerRead uint32

	//
	AvgsecPerRead_Base uint32

	//
	AvgsecPerWrite uint32

	//
	AvgsecPerWrite_Base uint32

	//
	AvgWriteQueueLength uint64

	//
	CreateFile uint64

	//
	CreateFilePersec uint64

	//
	CurrentReadQueueLength uint64

	//
	CurrentWriteQueueLength uint64

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
	MetadataIO uint64

	//
	MetadataIOPersec uint64

	//
	ReadBytesPersec uint64

	//
	Reads uint64

	//
	ReadsPersec uint64

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
	WriteBytesPersec uint64

	//
	Writes uint64

	//
	WritesPersec uint64
}

func NewWin32_PerfRawData_CsvFsPerfProvider_ClusterCSVFSEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_CsvFsPerfProvider_ClusterCSVFSEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvgBytesPerRead sets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerRead", (value))
}

// GetAvgBytesPerRead gets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgBytesPerRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerRead")
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

// SetAvgBytesPerRead_Base sets the value of AvgBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgBytesPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgBytesPerRead_Base", (value))
}

// GetAvgBytesPerRead_Base gets the value of AvgBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgBytesPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerRead_Base")
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

// SetAvgBytesPerWrite sets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerWrite", (value))
}

// GetAvgBytesPerWrite gets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgBytesPerWrite() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerWrite")
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

// SetAvgBytesPerWrite_Base sets the value of AvgBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgBytesPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgBytesPerWrite_Base", (value))
}

// GetAvgBytesPerWrite_Base gets the value of AvgBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgBytesPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerWrite_Base")
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

// SetAvgReadQueueLength sets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgReadQueueLength", (value))
}

// GetAvgReadQueueLength gets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgReadQueueLength")
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

// SetAvgsecPerRead sets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgsecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead", (value))
}

// GetAvgsecPerRead gets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgsecPerRead() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRead")
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

// SetAvgsecPerRead_Base sets the value of AvgsecPerRead_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgsecPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead_Base", (value))
}

// GetAvgsecPerRead_Base gets the value of AvgsecPerRead_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgsecPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRead_Base")
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

// SetAvgsecPerWrite sets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgsecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite", (value))
}

// GetAvgsecPerWrite gets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgsecPerWrite() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerWrite")
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

// SetAvgsecPerWrite_Base sets the value of AvgsecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgsecPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite_Base", (value))
}

// GetAvgsecPerWrite_Base gets the value of AvgsecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgsecPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerWrite_Base")
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

// SetAvgWriteQueueLength sets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyAvgWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgWriteQueueLength", (value))
}

// GetAvgWriteQueueLength gets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyAvgWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgWriteQueueLength")
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

// SetCreateFile sets the value of CreateFile for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyCreateFile(value uint64) (err error) {
	return instance.SetProperty("CreateFile", (value))
}

// GetCreateFile gets the value of CreateFile for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyCreateFile() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyCreateFilePersec(value uint64) (err error) {
	return instance.SetProperty("CreateFilePersec", (value))
}

// GetCreateFilePersec gets the value of CreateFilePersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyCreateFilePersec() (value uint64, err error) {
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

// SetCurrentReadQueueLength sets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyCurrentReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentReadQueueLength", (value))
}

// GetCurrentReadQueueLength gets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyCurrentReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentReadQueueLength")
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

// SetCurrentWriteQueueLength sets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyCurrentWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentWriteQueueLength", (value))
}

// GetCurrentWriteQueueLength gets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyCurrentWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentWriteQueueLength")
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyFilesInvalidatedDuringResume(value uint64) (err error) {
	return instance.SetProperty("FilesInvalidatedDuringResume", (value))
}

// GetFilesInvalidatedDuringResume gets the value of FilesInvalidatedDuringResume for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyFilesInvalidatedDuringResume() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyFilesInvalidatedOther(value uint64) (err error) {
	return instance.SetProperty("FilesInvalidatedOther", (value))
}

// GetFilesInvalidatedOther gets the value of FilesInvalidatedOther for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyFilesInvalidatedOther() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyFilesOpened(value uint32) (err error) {
	return instance.SetProperty("FilesOpened", (value))
}

// GetFilesOpened gets the value of FilesOpened for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyFilesOpened() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyFlushes(value uint64) (err error) {
	return instance.SetProperty("Flushes", (value))
}

// GetFlushes gets the value of Flushes for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyFlushes() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyFlushesPersec(value uint64) (err error) {
	return instance.SetProperty("FlushesPersec", (value))
}

// GetFlushesPersec gets the value of FlushesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyFlushesPersec() (value uint64, err error) {
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

// SetMetadataIO sets the value of MetadataIO for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyMetadataIO(value uint64) (err error) {
	return instance.SetProperty("MetadataIO", (value))
}

// GetMetadataIO gets the value of MetadataIO for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyMetadataIO() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyMetadataIOPersec(value uint64) (err error) {
	return instance.SetProperty("MetadataIOPersec", (value))
}

// GetMetadataIOPersec gets the value of MetadataIOPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyMetadataIOPersec() (value uint64, err error) {
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

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReads sets the value of Reads for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyReads(value uint64) (err error) {
	return instance.SetProperty("Reads", (value))
}

// GetReads gets the value of Reads for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyReads() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyReadsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec", (value))
}

// GetReadsPersec gets the value of ReadsPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyReadsPersec() (value uint64, err error) {
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

// SetVolumePauseCountDisk sets the value of VolumePauseCountDisk for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyVolumePauseCountDisk(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountDisk", (value))
}

// GetVolumePauseCountDisk gets the value of VolumePauseCountDisk for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyVolumePauseCountDisk() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyVolumePauseCountNetwork(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountNetwork", (value))
}

// GetVolumePauseCountNetwork gets the value of VolumePauseCountNetwork for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyVolumePauseCountNetwork() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyVolumePauseCountOther(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountOther", (value))
}

// GetVolumePauseCountOther gets the value of VolumePauseCountOther for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyVolumePauseCountOther() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyVolumePauseCountTotal(value uint64) (err error) {
	return instance.SetProperty("VolumePauseCountTotal", (value))
}

// GetVolumePauseCountTotal gets the value of VolumePauseCountTotal for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyVolumePauseCountTotal() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyVolumeState(value uint32) (err error) {
	return instance.SetProperty("VolumeState", (value))
}

// GetVolumeState gets the value of VolumeState for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyVolumeState() (value uint32, err error) {
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

// SetWriteBytesPersec sets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersec")
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyWrites(value uint64) (err error) {
	return instance.SetProperty("Writes", (value))
}

// GetWrites gets the value of Writes for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyWrites() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) SetPropertyWritesPersec(value uint64) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfRawData_CsvFsPerfProvider_ClusterCSVFS) GetPropertyWritesPersec() (value uint64, err error) {
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
