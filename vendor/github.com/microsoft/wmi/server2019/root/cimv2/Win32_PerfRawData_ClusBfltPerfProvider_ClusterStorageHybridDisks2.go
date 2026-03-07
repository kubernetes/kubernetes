// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 3/19/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2 struct
type Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2 struct {
	*Win32_PerfRawData

	//
	HeatMapFreeMemory uint64

	//
	HeatMapWindow uint64

	//
	RateDiskVRCReads uint64

	//
	RateDiskVRCReads_Base uint32

	//
	VRCHitReadBytes uint64

	//
	VRCHitReadBytesPersec uint64

	//
	VRCHitReads uint64

	//
	VRCHitReadsPersec uint64

	//
	VRCPopulateBytes uint64

	//
	VRCPopulateBytesPersec uint64

	//
	VRCPopulates uint64

	//
	VRCPopulatesPersec uint64
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetHeatMapFreeMemory sets the value of HeatMapFreeMemory for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyHeatMapFreeMemory(value uint64) (err error) {
	return instance.SetProperty("HeatMapFreeMemory", value)
}

// GetHeatMapFreeMemory gets the value of HeatMapFreeMemory for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyHeatMapFreeMemory() (value uint64, err error) {
	retValue, err := instance.GetProperty("HeatMapFreeMemory")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetHeatMapWindow sets the value of HeatMapWindow for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyHeatMapWindow(value uint64) (err error) {
	return instance.SetProperty("HeatMapWindow", value)
}

// GetHeatMapWindow gets the value of HeatMapWindow for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyHeatMapWindow() (value uint64, err error) {
	retValue, err := instance.GetProperty("HeatMapWindow")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetRateDiskVRCReads sets the value of RateDiskVRCReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyRateDiskVRCReads(value uint64) (err error) {
	return instance.SetProperty("RateDiskVRCReads", value)
}

// GetRateDiskVRCReads gets the value of RateDiskVRCReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyRateDiskVRCReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("RateDiskVRCReads")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetRateDiskVRCReads_Base sets the value of RateDiskVRCReads_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyRateDiskVRCReads_Base(value uint32) (err error) {
	return instance.SetProperty("RateDiskVRCReads_Base", value)
}

// GetRateDiskVRCReads_Base gets the value of RateDiskVRCReads_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyRateDiskVRCReads_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("RateDiskVRCReads_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCHitReadBytes sets the value of VRCHitReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCHitReadBytes(value uint64) (err error) {
	return instance.SetProperty("VRCHitReadBytes", value)
}

// GetVRCHitReadBytes gets the value of VRCHitReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCHitReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCHitReadBytes")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCHitReadBytesPersec sets the value of VRCHitReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCHitReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("VRCHitReadBytesPersec", value)
}

// GetVRCHitReadBytesPersec gets the value of VRCHitReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCHitReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCHitReadBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCHitReads sets the value of VRCHitReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCHitReads(value uint64) (err error) {
	return instance.SetProperty("VRCHitReads", value)
}

// GetVRCHitReads gets the value of VRCHitReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCHitReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCHitReads")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCHitReadsPersec sets the value of VRCHitReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCHitReadsPersec(value uint64) (err error) {
	return instance.SetProperty("VRCHitReadsPersec", value)
}

// GetVRCHitReadsPersec gets the value of VRCHitReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCHitReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCHitReadsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCPopulateBytes sets the value of VRCPopulateBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCPopulateBytes(value uint64) (err error) {
	return instance.SetProperty("VRCPopulateBytes", value)
}

// GetVRCPopulateBytes gets the value of VRCPopulateBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCPopulateBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCPopulateBytes")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCPopulateBytesPersec sets the value of VRCPopulateBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCPopulateBytesPersec(value uint64) (err error) {
	return instance.SetProperty("VRCPopulateBytesPersec", value)
}

// GetVRCPopulateBytesPersec gets the value of VRCPopulateBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCPopulateBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCPopulateBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCPopulates sets the value of VRCPopulates for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCPopulates(value uint64) (err error) {
	return instance.SetProperty("VRCPopulates", value)
}

// GetVRCPopulates gets the value of VRCPopulates for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCPopulates() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCPopulates")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetVRCPopulatesPersec sets the value of VRCPopulatesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) SetPropertyVRCPopulatesPersec(value uint64) (err error) {
	return instance.SetProperty("VRCPopulatesPersec", value)
}

// GetVRCPopulatesPersec gets the value of VRCPopulatesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks2) GetPropertyVRCPopulatesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("VRCPopulatesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}
