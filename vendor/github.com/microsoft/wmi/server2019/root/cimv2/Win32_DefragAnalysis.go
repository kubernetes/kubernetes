// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_DefragAnalysis struct
type Win32_DefragAnalysis struct {
	*cim.WmiInstance

	//
	AverageFileSize uint64

	//
	AverageFragmentsPerFile float64

	//
	AverageFreeSpacePerExtent uint64

	//
	ClusterSize uint64

	//
	ExcessFolderFragments uint64

	//
	FilePercentFragmentation uint32

	//
	FragmentedFolders uint64

	//
	FreeSpace uint64

	//
	FreeSpacePercent uint32

	//
	FreeSpacePercentFragmentation uint32

	//
	LargestFreeSpaceExtent uint64

	//
	MFTPercentInUse uint32

	//
	MFTRecordCount uint64

	//
	PageFileSize uint64

	//
	TotalExcessFragments uint64

	//
	TotalFiles uint64

	//
	TotalFolders uint64

	//
	TotalFragmentedFiles uint64

	//
	TotalFreeSpaceExtents uint64

	//
	TotalMFTFragments uint64

	//
	TotalMFTSize uint64

	//
	TotalPageFileFragments uint64

	//
	TotalPercentFragmentation uint32

	//
	TotalUnmovableFiles uint64

	//
	UsedSpace uint64

	//
	VolumeName string

	//
	VolumeSize uint64
}

func NewWin32_DefragAnalysisEx1(instance *cim.WmiInstance) (newInstance *Win32_DefragAnalysis, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_DefragAnalysis{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_DefragAnalysisEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DefragAnalysis, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DefragAnalysis{
		WmiInstance: tmp,
	}
	return
}

// SetAverageFileSize sets the value of AverageFileSize for the instance
func (instance *Win32_DefragAnalysis) SetPropertyAverageFileSize(value uint64) (err error) {
	return instance.SetProperty("AverageFileSize", (value))
}

// GetAverageFileSize gets the value of AverageFileSize for the instance
func (instance *Win32_DefragAnalysis) GetPropertyAverageFileSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageFileSize")
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

// SetAverageFragmentsPerFile sets the value of AverageFragmentsPerFile for the instance
func (instance *Win32_DefragAnalysis) SetPropertyAverageFragmentsPerFile(value float64) (err error) {
	return instance.SetProperty("AverageFragmentsPerFile", (value))
}

// GetAverageFragmentsPerFile gets the value of AverageFragmentsPerFile for the instance
func (instance *Win32_DefragAnalysis) GetPropertyAverageFragmentsPerFile() (value float64, err error) {
	retValue, err := instance.GetProperty("AverageFragmentsPerFile")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}

// SetAverageFreeSpacePerExtent sets the value of AverageFreeSpacePerExtent for the instance
func (instance *Win32_DefragAnalysis) SetPropertyAverageFreeSpacePerExtent(value uint64) (err error) {
	return instance.SetProperty("AverageFreeSpacePerExtent", (value))
}

// GetAverageFreeSpacePerExtent gets the value of AverageFreeSpacePerExtent for the instance
func (instance *Win32_DefragAnalysis) GetPropertyAverageFreeSpacePerExtent() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageFreeSpacePerExtent")
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

// SetClusterSize sets the value of ClusterSize for the instance
func (instance *Win32_DefragAnalysis) SetPropertyClusterSize(value uint64) (err error) {
	return instance.SetProperty("ClusterSize", (value))
}

// GetClusterSize gets the value of ClusterSize for the instance
func (instance *Win32_DefragAnalysis) GetPropertyClusterSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("ClusterSize")
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

// SetExcessFolderFragments sets the value of ExcessFolderFragments for the instance
func (instance *Win32_DefragAnalysis) SetPropertyExcessFolderFragments(value uint64) (err error) {
	return instance.SetProperty("ExcessFolderFragments", (value))
}

// GetExcessFolderFragments gets the value of ExcessFolderFragments for the instance
func (instance *Win32_DefragAnalysis) GetPropertyExcessFolderFragments() (value uint64, err error) {
	retValue, err := instance.GetProperty("ExcessFolderFragments")
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

// SetFilePercentFragmentation sets the value of FilePercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) SetPropertyFilePercentFragmentation(value uint32) (err error) {
	return instance.SetProperty("FilePercentFragmentation", (value))
}

// GetFilePercentFragmentation gets the value of FilePercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) GetPropertyFilePercentFragmentation() (value uint32, err error) {
	retValue, err := instance.GetProperty("FilePercentFragmentation")
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

// SetFragmentedFolders sets the value of FragmentedFolders for the instance
func (instance *Win32_DefragAnalysis) SetPropertyFragmentedFolders(value uint64) (err error) {
	return instance.SetProperty("FragmentedFolders", (value))
}

// GetFragmentedFolders gets the value of FragmentedFolders for the instance
func (instance *Win32_DefragAnalysis) GetPropertyFragmentedFolders() (value uint64, err error) {
	retValue, err := instance.GetProperty("FragmentedFolders")
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

// SetFreeSpace sets the value of FreeSpace for the instance
func (instance *Win32_DefragAnalysis) SetPropertyFreeSpace(value uint64) (err error) {
	return instance.SetProperty("FreeSpace", (value))
}

// GetFreeSpace gets the value of FreeSpace for the instance
func (instance *Win32_DefragAnalysis) GetPropertyFreeSpace() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreeSpace")
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

// SetFreeSpacePercent sets the value of FreeSpacePercent for the instance
func (instance *Win32_DefragAnalysis) SetPropertyFreeSpacePercent(value uint32) (err error) {
	return instance.SetProperty("FreeSpacePercent", (value))
}

// GetFreeSpacePercent gets the value of FreeSpacePercent for the instance
func (instance *Win32_DefragAnalysis) GetPropertyFreeSpacePercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeSpacePercent")
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

// SetFreeSpacePercentFragmentation sets the value of FreeSpacePercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) SetPropertyFreeSpacePercentFragmentation(value uint32) (err error) {
	return instance.SetProperty("FreeSpacePercentFragmentation", (value))
}

// GetFreeSpacePercentFragmentation gets the value of FreeSpacePercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) GetPropertyFreeSpacePercentFragmentation() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeSpacePercentFragmentation")
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

// SetLargestFreeSpaceExtent sets the value of LargestFreeSpaceExtent for the instance
func (instance *Win32_DefragAnalysis) SetPropertyLargestFreeSpaceExtent(value uint64) (err error) {
	return instance.SetProperty("LargestFreeSpaceExtent", (value))
}

// GetLargestFreeSpaceExtent gets the value of LargestFreeSpaceExtent for the instance
func (instance *Win32_DefragAnalysis) GetPropertyLargestFreeSpaceExtent() (value uint64, err error) {
	retValue, err := instance.GetProperty("LargestFreeSpaceExtent")
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

// SetMFTPercentInUse sets the value of MFTPercentInUse for the instance
func (instance *Win32_DefragAnalysis) SetPropertyMFTPercentInUse(value uint32) (err error) {
	return instance.SetProperty("MFTPercentInUse", (value))
}

// GetMFTPercentInUse gets the value of MFTPercentInUse for the instance
func (instance *Win32_DefragAnalysis) GetPropertyMFTPercentInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("MFTPercentInUse")
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

// SetMFTRecordCount sets the value of MFTRecordCount for the instance
func (instance *Win32_DefragAnalysis) SetPropertyMFTRecordCount(value uint64) (err error) {
	return instance.SetProperty("MFTRecordCount", (value))
}

// GetMFTRecordCount gets the value of MFTRecordCount for the instance
func (instance *Win32_DefragAnalysis) GetPropertyMFTRecordCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("MFTRecordCount")
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

// SetPageFileSize sets the value of PageFileSize for the instance
func (instance *Win32_DefragAnalysis) SetPropertyPageFileSize(value uint64) (err error) {
	return instance.SetProperty("PageFileSize", (value))
}

// GetPageFileSize gets the value of PageFileSize for the instance
func (instance *Win32_DefragAnalysis) GetPropertyPageFileSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageFileSize")
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

// SetTotalExcessFragments sets the value of TotalExcessFragments for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalExcessFragments(value uint64) (err error) {
	return instance.SetProperty("TotalExcessFragments", (value))
}

// GetTotalExcessFragments gets the value of TotalExcessFragments for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalExcessFragments() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalExcessFragments")
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

// SetTotalFiles sets the value of TotalFiles for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalFiles(value uint64) (err error) {
	return instance.SetProperty("TotalFiles", (value))
}

// GetTotalFiles gets the value of TotalFiles for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalFiles() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalFiles")
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

// SetTotalFolders sets the value of TotalFolders for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalFolders(value uint64) (err error) {
	return instance.SetProperty("TotalFolders", (value))
}

// GetTotalFolders gets the value of TotalFolders for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalFolders() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalFolders")
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

// SetTotalFragmentedFiles sets the value of TotalFragmentedFiles for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalFragmentedFiles(value uint64) (err error) {
	return instance.SetProperty("TotalFragmentedFiles", (value))
}

// GetTotalFragmentedFiles gets the value of TotalFragmentedFiles for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalFragmentedFiles() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalFragmentedFiles")
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

// SetTotalFreeSpaceExtents sets the value of TotalFreeSpaceExtents for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalFreeSpaceExtents(value uint64) (err error) {
	return instance.SetProperty("TotalFreeSpaceExtents", (value))
}

// GetTotalFreeSpaceExtents gets the value of TotalFreeSpaceExtents for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalFreeSpaceExtents() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalFreeSpaceExtents")
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

// SetTotalMFTFragments sets the value of TotalMFTFragments for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalMFTFragments(value uint64) (err error) {
	return instance.SetProperty("TotalMFTFragments", (value))
}

// GetTotalMFTFragments gets the value of TotalMFTFragments for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalMFTFragments() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalMFTFragments")
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

// SetTotalMFTSize sets the value of TotalMFTSize for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalMFTSize(value uint64) (err error) {
	return instance.SetProperty("TotalMFTSize", (value))
}

// GetTotalMFTSize gets the value of TotalMFTSize for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalMFTSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalMFTSize")
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

// SetTotalPageFileFragments sets the value of TotalPageFileFragments for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalPageFileFragments(value uint64) (err error) {
	return instance.SetProperty("TotalPageFileFragments", (value))
}

// GetTotalPageFileFragments gets the value of TotalPageFileFragments for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalPageFileFragments() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalPageFileFragments")
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

// SetTotalPercentFragmentation sets the value of TotalPercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalPercentFragmentation(value uint32) (err error) {
	return instance.SetProperty("TotalPercentFragmentation", (value))
}

// GetTotalPercentFragmentation gets the value of TotalPercentFragmentation for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalPercentFragmentation() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalPercentFragmentation")
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

// SetTotalUnmovableFiles sets the value of TotalUnmovableFiles for the instance
func (instance *Win32_DefragAnalysis) SetPropertyTotalUnmovableFiles(value uint64) (err error) {
	return instance.SetProperty("TotalUnmovableFiles", (value))
}

// GetTotalUnmovableFiles gets the value of TotalUnmovableFiles for the instance
func (instance *Win32_DefragAnalysis) GetPropertyTotalUnmovableFiles() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalUnmovableFiles")
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

// SetUsedSpace sets the value of UsedSpace for the instance
func (instance *Win32_DefragAnalysis) SetPropertyUsedSpace(value uint64) (err error) {
	return instance.SetProperty("UsedSpace", (value))
}

// GetUsedSpace gets the value of UsedSpace for the instance
func (instance *Win32_DefragAnalysis) GetPropertyUsedSpace() (value uint64, err error) {
	retValue, err := instance.GetProperty("UsedSpace")
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

// SetVolumeName sets the value of VolumeName for the instance
func (instance *Win32_DefragAnalysis) SetPropertyVolumeName(value string) (err error) {
	return instance.SetProperty("VolumeName", (value))
}

// GetVolumeName gets the value of VolumeName for the instance
func (instance *Win32_DefragAnalysis) GetPropertyVolumeName() (value string, err error) {
	retValue, err := instance.GetProperty("VolumeName")
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

// SetVolumeSize sets the value of VolumeSize for the instance
func (instance *Win32_DefragAnalysis) SetPropertyVolumeSize(value uint64) (err error) {
	return instance.SetProperty("VolumeSize", (value))
}

// GetVolumeSize gets the value of VolumeSize for the instance
func (instance *Win32_DefragAnalysis) GetPropertyVolumeSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("VolumeSize")
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
