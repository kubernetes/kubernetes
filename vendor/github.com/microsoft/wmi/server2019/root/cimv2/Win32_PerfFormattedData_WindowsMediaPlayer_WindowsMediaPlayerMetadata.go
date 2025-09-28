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

// Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata struct
type Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata struct {
	*Win32_PerfFormattedData

	//
	AFTSExecutionTimems uint32

	//
	ArtExtractionTimems uint32

	//
	CommitTimems uint32

	//
	DirectoryChangeQueueLength uint32

	//
	DirtyDirectoryHitCount uint32

	//
	FileScanningThreadPrioirty uint32

	//
	FilesScannedPerMinute uint64

	//
	GrovelerServiceRoutineExecutionsPerSecond uint64

	//
	LibraryDescriptionChangeNotificationsPerSecond uint64

	//
	LibraryDescriptionUpdatesPerSecond uint64

	//
	MonitoredFolderUpdatesPerSecond uint64

	//
	NormalizationTimems uint32

	//
	PropertyExtractionTimems uint32

	//
	ReorganizeTimems uint32

	//
	ScanningState uint32

	//
	TimestampDirectoryHitCount uint32

	//
	URLClassificationTimems uint32
}

func NewWin32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadataEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadataEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAFTSExecutionTimems sets the value of AFTSExecutionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyAFTSExecutionTimems(value uint32) (err error) {
	return instance.SetProperty("AFTSExecutionTimems", (value))
}

// GetAFTSExecutionTimems gets the value of AFTSExecutionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyAFTSExecutionTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("AFTSExecutionTimems")
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

// SetArtExtractionTimems sets the value of ArtExtractionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyArtExtractionTimems(value uint32) (err error) {
	return instance.SetProperty("ArtExtractionTimems", (value))
}

// GetArtExtractionTimems gets the value of ArtExtractionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyArtExtractionTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("ArtExtractionTimems")
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

// SetCommitTimems sets the value of CommitTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyCommitTimems(value uint32) (err error) {
	return instance.SetProperty("CommitTimems", (value))
}

// GetCommitTimems gets the value of CommitTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyCommitTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("CommitTimems")
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

// SetDirectoryChangeQueueLength sets the value of DirectoryChangeQueueLength for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyDirectoryChangeQueueLength(value uint32) (err error) {
	return instance.SetProperty("DirectoryChangeQueueLength", (value))
}

// GetDirectoryChangeQueueLength gets the value of DirectoryChangeQueueLength for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyDirectoryChangeQueueLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("DirectoryChangeQueueLength")
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

// SetDirtyDirectoryHitCount sets the value of DirtyDirectoryHitCount for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyDirtyDirectoryHitCount(value uint32) (err error) {
	return instance.SetProperty("DirtyDirectoryHitCount", (value))
}

// GetDirtyDirectoryHitCount gets the value of DirtyDirectoryHitCount for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyDirtyDirectoryHitCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("DirtyDirectoryHitCount")
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

// SetFileScanningThreadPrioirty sets the value of FileScanningThreadPrioirty for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyFileScanningThreadPrioirty(value uint32) (err error) {
	return instance.SetProperty("FileScanningThreadPrioirty", (value))
}

// GetFileScanningThreadPrioirty gets the value of FileScanningThreadPrioirty for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyFileScanningThreadPrioirty() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileScanningThreadPrioirty")
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

// SetFilesScannedPerMinute sets the value of FilesScannedPerMinute for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyFilesScannedPerMinute(value uint64) (err error) {
	return instance.SetProperty("FilesScannedPerMinute", (value))
}

// GetFilesScannedPerMinute gets the value of FilesScannedPerMinute for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyFilesScannedPerMinute() (value uint64, err error) {
	retValue, err := instance.GetProperty("FilesScannedPerMinute")
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

// SetGrovelerServiceRoutineExecutionsPerSecond sets the value of GrovelerServiceRoutineExecutionsPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyGrovelerServiceRoutineExecutionsPerSecond(value uint64) (err error) {
	return instance.SetProperty("GrovelerServiceRoutineExecutionsPerSecond", (value))
}

// GetGrovelerServiceRoutineExecutionsPerSecond gets the value of GrovelerServiceRoutineExecutionsPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyGrovelerServiceRoutineExecutionsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("GrovelerServiceRoutineExecutionsPerSecond")
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

// SetLibraryDescriptionChangeNotificationsPerSecond sets the value of LibraryDescriptionChangeNotificationsPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyLibraryDescriptionChangeNotificationsPerSecond(value uint64) (err error) {
	return instance.SetProperty("LibraryDescriptionChangeNotificationsPerSecond", (value))
}

// GetLibraryDescriptionChangeNotificationsPerSecond gets the value of LibraryDescriptionChangeNotificationsPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyLibraryDescriptionChangeNotificationsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("LibraryDescriptionChangeNotificationsPerSecond")
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

// SetLibraryDescriptionUpdatesPerSecond sets the value of LibraryDescriptionUpdatesPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyLibraryDescriptionUpdatesPerSecond(value uint64) (err error) {
	return instance.SetProperty("LibraryDescriptionUpdatesPerSecond", (value))
}

// GetLibraryDescriptionUpdatesPerSecond gets the value of LibraryDescriptionUpdatesPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyLibraryDescriptionUpdatesPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("LibraryDescriptionUpdatesPerSecond")
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

// SetMonitoredFolderUpdatesPerSecond sets the value of MonitoredFolderUpdatesPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyMonitoredFolderUpdatesPerSecond(value uint64) (err error) {
	return instance.SetProperty("MonitoredFolderUpdatesPerSecond", (value))
}

// GetMonitoredFolderUpdatesPerSecond gets the value of MonitoredFolderUpdatesPerSecond for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyMonitoredFolderUpdatesPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("MonitoredFolderUpdatesPerSecond")
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

// SetNormalizationTimems sets the value of NormalizationTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyNormalizationTimems(value uint32) (err error) {
	return instance.SetProperty("NormalizationTimems", (value))
}

// GetNormalizationTimems gets the value of NormalizationTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyNormalizationTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("NormalizationTimems")
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

// SetPropertyExtractionTimems sets the value of PropertyExtractionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyPropertyExtractionTimems(value uint32) (err error) {
	return instance.SetProperty("PropertyExtractionTimems", (value))
}

// GetPropertyExtractionTimems gets the value of PropertyExtractionTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyPropertyExtractionTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("PropertyExtractionTimems")
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

// SetReorganizeTimems sets the value of ReorganizeTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyReorganizeTimems(value uint32) (err error) {
	return instance.SetProperty("ReorganizeTimems", (value))
}

// GetReorganizeTimems gets the value of ReorganizeTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyReorganizeTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReorganizeTimems")
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

// SetScanningState sets the value of ScanningState for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyScanningState(value uint32) (err error) {
	return instance.SetProperty("ScanningState", (value))
}

// GetScanningState gets the value of ScanningState for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyScanningState() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScanningState")
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

// SetTimestampDirectoryHitCount sets the value of TimestampDirectoryHitCount for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyTimestampDirectoryHitCount(value uint32) (err error) {
	return instance.SetProperty("TimestampDirectoryHitCount", (value))
}

// GetTimestampDirectoryHitCount gets the value of TimestampDirectoryHitCount for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyTimestampDirectoryHitCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimestampDirectoryHitCount")
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

// SetURLClassificationTimems sets the value of URLClassificationTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) SetPropertyURLClassificationTimems(value uint32) (err error) {
	return instance.SetProperty("URLClassificationTimems", (value))
}

// GetURLClassificationTimems gets the value of URLClassificationTimems for the instance
func (instance *Win32_PerfFormattedData_WindowsMediaPlayer_WindowsMediaPlayerMetadata) GetPropertyURLClassificationTimems() (value uint32, err error) {
	retValue, err := instance.GetProperty("URLClassificationTimems")
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
