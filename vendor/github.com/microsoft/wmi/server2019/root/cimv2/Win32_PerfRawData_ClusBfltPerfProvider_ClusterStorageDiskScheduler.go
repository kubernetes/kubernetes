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

// Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler struct
type Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler struct {
	*Win32_PerfRawData

	//
	DspPerSysAvgQueueLength uint64

	//
	DspPerSysHighAvgQueueLength uint64

	//
	DspPerSysHighAvgsecPerDataRequest uint32

	//
	DspPerSysHighAvgsecPerDataRequest_Base uint32

	//
	DspPerSysHighCurrentQueueLength uint64

	//
	DspPerSysIdlePerLowAvgQueueLength uint64

	//
	DspPerSysIdlePerLowAvgsecPerDataRequest uint32

	//
	DspPerSysIdlePerLowAvgsecPerDataRequest_Base uint32

	//
	DspPerSysIdlePerLowCurrentQueueLength uint64

	//
	DspPerSysNormalAvgQueueLength uint64

	//
	DspPerSysNormalAvgsecPerDataRequest uint32

	//
	DspPerSysNormalAvgsecPerDataRequest_Base uint32

	//
	DspPerSysNormalCurrentQueueLength uint64

	//
	DspPerUsrAvgQueueLength uint64

	//
	DspPerUsrHighAvgQueueLength uint64

	//
	DspPerUsrHighAvgsecPerDataRequest uint32

	//
	DspPerUsrHighAvgsecPerDataRequest_Base uint32

	//
	DspPerUsrHighCurrentQueueLength uint64

	//
	DspPerUsrIdlePerLowAvgQueueLength uint64

	//
	DspPerUsrIdlePerLowAvgsecPerDataRequest uint32

	//
	DspPerUsrIdlePerLowAvgsecPerDataRequest_Base uint32

	//
	DspPerUsrIdlePerLowCurrentQueueLength uint64

	//
	DspPerUsrNormalAvgQueueLength uint64

	//
	DspPerUsrNormalAvgsecPerDataRequest uint32

	//
	DspPerUsrNormalAvgsecPerDataRequest_Base uint32

	//
	DspPerUsrNormalCurrentQueueLength uint64

	//
	QuePerSysAvgQueueLength uint64

	//
	QuePerSysHighAvgQueueLength uint64

	//
	QuePerSysHighAvgsecPerDataRequest uint32

	//
	QuePerSysHighAvgsecPerDataRequest_Base uint32

	//
	QuePerSysHighBytesPersec uint64

	//
	QuePerSysHighCurrentQueueLength uint64

	//
	QuePerSysHighDataRequestsPersec uint64

	//
	QuePerSysIdlePerLowAvgQueueLength uint64

	//
	QuePerSysIdlePerLowAvgsecPerDataRequest uint32

	//
	QuePerSysIdlePerLowAvgsecPerDataRequest_Base uint32

	//
	QuePerSysIdlePerLowBytesPersec uint64

	//
	QuePerSysIdlePerLowCurrentQueueLength uint64

	//
	QuePerSysIdlePerLowDataRequestsPersec uint64

	//
	QuePerSysNormalAvgQueueLength uint64

	//
	QuePerSysNormalAvgsecPerDataRequest uint32

	//
	QuePerSysNormalAvgsecPerDataRequest_Base uint32

	//
	QuePerSysNormalBytesPersec uint64

	//
	QuePerSysNormalCurrentQueueLength uint64

	//
	QuePerSysNormalDataRequestsPersec uint64

	//
	QuePerUsrAvgQueueLength uint64

	//
	QuePerUsrHighAvgQueueLength uint64

	//
	QuePerUsrHighAvgsecPerDataRequest uint32

	//
	QuePerUsrHighAvgsecPerDataRequest_Base uint32

	//
	QuePerUsrHighBytesPersec uint64

	//
	QuePerUsrHighCurrentQueueLength uint64

	//
	QuePerUsrHighDataRequestsPersec uint64

	//
	QuePerUsrIdlePerLowAvgQueueLength uint64

	//
	QuePerUsrIdlePerLowAvgsecPerDataRequest uint32

	//
	QuePerUsrIdlePerLowAvgsecPerDataRequest_Base uint32

	//
	QuePerUsrIdlePerLowBytesPersec uint64

	//
	QuePerUsrIdlePerLowCurrentQueueLength uint64

	//
	QuePerUsrIdlePerLowDataRequestsPersec uint64

	//
	QuePerUsrNormalAvgQueueLength uint64

	//
	QuePerUsrNormalAvgsecPerDataRequest uint32

	//
	QuePerUsrNormalAvgsecPerDataRequest_Base uint32

	//
	QuePerUsrNormalBytesPersec uint64

	//
	QuePerUsrNormalCurrentQueueLength uint64

	//
	QuePerUsrNormalDataRequestsPersec uint64
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskSchedulerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskSchedulerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDspPerSysAvgQueueLength sets the value of DspPerSysAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysAvgQueueLength", value)
}

// GetDspPerSysAvgQueueLength gets the value of DspPerSysAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysHighAvgQueueLength sets the value of DspPerSysHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysHighAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysHighAvgQueueLength", value)
}

// GetDspPerSysHighAvgQueueLength gets the value of DspPerSysHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysHighAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysHighAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysHighAvgsecPerDataRequest sets the value of DspPerSysHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysHighAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerSysHighAvgsecPerDataRequest", value)
}

// GetDspPerSysHighAvgsecPerDataRequest gets the value of DspPerSysHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysHighAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysHighAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysHighAvgsecPerDataRequest_Base sets the value of DspPerSysHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysHighAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerSysHighAvgsecPerDataRequest_Base", value)
}

// GetDspPerSysHighAvgsecPerDataRequest_Base gets the value of DspPerSysHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysHighAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysHighAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysHighCurrentQueueLength sets the value of DspPerSysHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysHighCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysHighCurrentQueueLength", value)
}

// GetDspPerSysHighCurrentQueueLength gets the value of DspPerSysHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysHighCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysHighCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysIdlePerLowAvgQueueLength sets the value of DspPerSysIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysIdlePerLowAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysIdlePerLowAvgQueueLength", value)
}

// GetDspPerSysIdlePerLowAvgQueueLength gets the value of DspPerSysIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysIdlePerLowAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysIdlePerLowAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysIdlePerLowAvgsecPerDataRequest sets the value of DspPerSysIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysIdlePerLowAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerSysIdlePerLowAvgsecPerDataRequest", value)
}

// GetDspPerSysIdlePerLowAvgsecPerDataRequest gets the value of DspPerSysIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysIdlePerLowAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysIdlePerLowAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysIdlePerLowAvgsecPerDataRequest_Base sets the value of DspPerSysIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysIdlePerLowAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerSysIdlePerLowAvgsecPerDataRequest_Base", value)
}

// GetDspPerSysIdlePerLowAvgsecPerDataRequest_Base gets the value of DspPerSysIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysIdlePerLowAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysIdlePerLowAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysIdlePerLowCurrentQueueLength sets the value of DspPerSysIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysIdlePerLowCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysIdlePerLowCurrentQueueLength", value)
}

// GetDspPerSysIdlePerLowCurrentQueueLength gets the value of DspPerSysIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysIdlePerLowCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysIdlePerLowCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysNormalAvgQueueLength sets the value of DspPerSysNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysNormalAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysNormalAvgQueueLength", value)
}

// GetDspPerSysNormalAvgQueueLength gets the value of DspPerSysNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysNormalAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysNormalAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysNormalAvgsecPerDataRequest sets the value of DspPerSysNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysNormalAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerSysNormalAvgsecPerDataRequest", value)
}

// GetDspPerSysNormalAvgsecPerDataRequest gets the value of DspPerSysNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysNormalAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysNormalAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysNormalAvgsecPerDataRequest_Base sets the value of DspPerSysNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysNormalAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerSysNormalAvgsecPerDataRequest_Base", value)
}

// GetDspPerSysNormalAvgsecPerDataRequest_Base gets the value of DspPerSysNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysNormalAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerSysNormalAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerSysNormalCurrentQueueLength sets the value of DspPerSysNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerSysNormalCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerSysNormalCurrentQueueLength", value)
}

// GetDspPerSysNormalCurrentQueueLength gets the value of DspPerSysNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerSysNormalCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerSysNormalCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrAvgQueueLength sets the value of DspPerUsrAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrAvgQueueLength", value)
}

// GetDspPerUsrAvgQueueLength gets the value of DspPerUsrAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrHighAvgQueueLength sets the value of DspPerUsrHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrHighAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrHighAvgQueueLength", value)
}

// GetDspPerUsrHighAvgQueueLength gets the value of DspPerUsrHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrHighAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrHighAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrHighAvgsecPerDataRequest sets the value of DspPerUsrHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrHighAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrHighAvgsecPerDataRequest", value)
}

// GetDspPerUsrHighAvgsecPerDataRequest gets the value of DspPerUsrHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrHighAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrHighAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrHighAvgsecPerDataRequest_Base sets the value of DspPerUsrHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrHighAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrHighAvgsecPerDataRequest_Base", value)
}

// GetDspPerUsrHighAvgsecPerDataRequest_Base gets the value of DspPerUsrHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrHighAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrHighAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrHighCurrentQueueLength sets the value of DspPerUsrHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrHighCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrHighCurrentQueueLength", value)
}

// GetDspPerUsrHighCurrentQueueLength gets the value of DspPerUsrHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrHighCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrHighCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrIdlePerLowAvgQueueLength sets the value of DspPerUsrIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrIdlePerLowAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrIdlePerLowAvgQueueLength", value)
}

// GetDspPerUsrIdlePerLowAvgQueueLength gets the value of DspPerUsrIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrIdlePerLowAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrIdlePerLowAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrIdlePerLowAvgsecPerDataRequest sets the value of DspPerUsrIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrIdlePerLowAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrIdlePerLowAvgsecPerDataRequest", value)
}

// GetDspPerUsrIdlePerLowAvgsecPerDataRequest gets the value of DspPerUsrIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrIdlePerLowAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrIdlePerLowAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrIdlePerLowAvgsecPerDataRequest_Base sets the value of DspPerUsrIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrIdlePerLowAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrIdlePerLowAvgsecPerDataRequest_Base", value)
}

// GetDspPerUsrIdlePerLowAvgsecPerDataRequest_Base gets the value of DspPerUsrIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrIdlePerLowAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrIdlePerLowAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrIdlePerLowCurrentQueueLength sets the value of DspPerUsrIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrIdlePerLowCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrIdlePerLowCurrentQueueLength", value)
}

// GetDspPerUsrIdlePerLowCurrentQueueLength gets the value of DspPerUsrIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrIdlePerLowCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrIdlePerLowCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrNormalAvgQueueLength sets the value of DspPerUsrNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrNormalAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrNormalAvgQueueLength", value)
}

// GetDspPerUsrNormalAvgQueueLength gets the value of DspPerUsrNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrNormalAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrNormalAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrNormalAvgsecPerDataRequest sets the value of DspPerUsrNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrNormalAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrNormalAvgsecPerDataRequest", value)
}

// GetDspPerUsrNormalAvgsecPerDataRequest gets the value of DspPerUsrNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrNormalAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrNormalAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrNormalAvgsecPerDataRequest_Base sets the value of DspPerUsrNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrNormalAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("DspPerUsrNormalAvgsecPerDataRequest_Base", value)
}

// GetDspPerUsrNormalAvgsecPerDataRequest_Base gets the value of DspPerUsrNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrNormalAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DspPerUsrNormalAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetDspPerUsrNormalCurrentQueueLength sets the value of DspPerUsrNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyDspPerUsrNormalCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("DspPerUsrNormalCurrentQueueLength", value)
}

// GetDspPerUsrNormalCurrentQueueLength gets the value of DspPerUsrNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyDspPerUsrNormalCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("DspPerUsrNormalCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysAvgQueueLength sets the value of QuePerSysAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysAvgQueueLength", value)
}

// GetQuePerSysAvgQueueLength gets the value of QuePerSysAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighAvgQueueLength sets the value of QuePerSysHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysHighAvgQueueLength", value)
}

// GetQuePerSysHighAvgQueueLength gets the value of QuePerSysHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighAvgsecPerDataRequest sets the value of QuePerSysHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerSysHighAvgsecPerDataRequest", value)
}

// GetQuePerSysHighAvgsecPerDataRequest gets the value of QuePerSysHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighAvgsecPerDataRequest_Base sets the value of QuePerSysHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerSysHighAvgsecPerDataRequest_Base", value)
}

// GetQuePerSysHighAvgsecPerDataRequest_Base gets the value of QuePerSysHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighBytesPersec sets the value of QuePerSysHighBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysHighBytesPersec", value)
}

// GetQuePerSysHighBytesPersec gets the value of QuePerSysHighBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighCurrentQueueLength sets the value of QuePerSysHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysHighCurrentQueueLength", value)
}

// GetQuePerSysHighCurrentQueueLength gets the value of QuePerSysHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysHighDataRequestsPersec sets the value of QuePerSysHighDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysHighDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysHighDataRequestsPersec", value)
}

// GetQuePerSysHighDataRequestsPersec gets the value of QuePerSysHighDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysHighDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysHighDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowAvgQueueLength sets the value of QuePerSysIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowAvgQueueLength", value)
}

// GetQuePerSysIdlePerLowAvgQueueLength gets the value of QuePerSysIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowAvgsecPerDataRequest sets the value of QuePerSysIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowAvgsecPerDataRequest", value)
}

// GetQuePerSysIdlePerLowAvgsecPerDataRequest gets the value of QuePerSysIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowAvgsecPerDataRequest_Base sets the value of QuePerSysIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowAvgsecPerDataRequest_Base", value)
}

// GetQuePerSysIdlePerLowAvgsecPerDataRequest_Base gets the value of QuePerSysIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowBytesPersec sets the value of QuePerSysIdlePerLowBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowBytesPersec", value)
}

// GetQuePerSysIdlePerLowBytesPersec gets the value of QuePerSysIdlePerLowBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowCurrentQueueLength sets the value of QuePerSysIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowCurrentQueueLength", value)
}

// GetQuePerSysIdlePerLowCurrentQueueLength gets the value of QuePerSysIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysIdlePerLowDataRequestsPersec sets the value of QuePerSysIdlePerLowDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysIdlePerLowDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysIdlePerLowDataRequestsPersec", value)
}

// GetQuePerSysIdlePerLowDataRequestsPersec gets the value of QuePerSysIdlePerLowDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysIdlePerLowDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysIdlePerLowDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalAvgQueueLength sets the value of QuePerSysNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysNormalAvgQueueLength", value)
}

// GetQuePerSysNormalAvgQueueLength gets the value of QuePerSysNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalAvgsecPerDataRequest sets the value of QuePerSysNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerSysNormalAvgsecPerDataRequest", value)
}

// GetQuePerSysNormalAvgsecPerDataRequest gets the value of QuePerSysNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalAvgsecPerDataRequest_Base sets the value of QuePerSysNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerSysNormalAvgsecPerDataRequest_Base", value)
}

// GetQuePerSysNormalAvgsecPerDataRequest_Base gets the value of QuePerSysNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalBytesPersec sets the value of QuePerSysNormalBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysNormalBytesPersec", value)
}

// GetQuePerSysNormalBytesPersec gets the value of QuePerSysNormalBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalCurrentQueueLength sets the value of QuePerSysNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerSysNormalCurrentQueueLength", value)
}

// GetQuePerSysNormalCurrentQueueLength gets the value of QuePerSysNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerSysNormalDataRequestsPersec sets the value of QuePerSysNormalDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerSysNormalDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerSysNormalDataRequestsPersec", value)
}

// GetQuePerSysNormalDataRequestsPersec gets the value of QuePerSysNormalDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerSysNormalDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerSysNormalDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrAvgQueueLength sets the value of QuePerUsrAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrAvgQueueLength", value)
}

// GetQuePerUsrAvgQueueLength gets the value of QuePerUsrAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighAvgQueueLength sets the value of QuePerUsrHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrHighAvgQueueLength", value)
}

// GetQuePerUsrHighAvgQueueLength gets the value of QuePerUsrHighAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighAvgsecPerDataRequest sets the value of QuePerUsrHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrHighAvgsecPerDataRequest", value)
}

// GetQuePerUsrHighAvgsecPerDataRequest gets the value of QuePerUsrHighAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighAvgsecPerDataRequest_Base sets the value of QuePerUsrHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrHighAvgsecPerDataRequest_Base", value)
}

// GetQuePerUsrHighAvgsecPerDataRequest_Base gets the value of QuePerUsrHighAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighBytesPersec sets the value of QuePerUsrHighBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrHighBytesPersec", value)
}

// GetQuePerUsrHighBytesPersec gets the value of QuePerUsrHighBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighCurrentQueueLength sets the value of QuePerUsrHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrHighCurrentQueueLength", value)
}

// GetQuePerUsrHighCurrentQueueLength gets the value of QuePerUsrHighCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrHighDataRequestsPersec sets the value of QuePerUsrHighDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrHighDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrHighDataRequestsPersec", value)
}

// GetQuePerUsrHighDataRequestsPersec gets the value of QuePerUsrHighDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrHighDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrHighDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowAvgQueueLength sets the value of QuePerUsrIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowAvgQueueLength", value)
}

// GetQuePerUsrIdlePerLowAvgQueueLength gets the value of QuePerUsrIdlePerLowAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowAvgsecPerDataRequest sets the value of QuePerUsrIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowAvgsecPerDataRequest", value)
}

// GetQuePerUsrIdlePerLowAvgsecPerDataRequest gets the value of QuePerUsrIdlePerLowAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowAvgsecPerDataRequest_Base sets the value of QuePerUsrIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowAvgsecPerDataRequest_Base", value)
}

// GetQuePerUsrIdlePerLowAvgsecPerDataRequest_Base gets the value of QuePerUsrIdlePerLowAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowBytesPersec sets the value of QuePerUsrIdlePerLowBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowBytesPersec", value)
}

// GetQuePerUsrIdlePerLowBytesPersec gets the value of QuePerUsrIdlePerLowBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowCurrentQueueLength sets the value of QuePerUsrIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowCurrentQueueLength", value)
}

// GetQuePerUsrIdlePerLowCurrentQueueLength gets the value of QuePerUsrIdlePerLowCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrIdlePerLowDataRequestsPersec sets the value of QuePerUsrIdlePerLowDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrIdlePerLowDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrIdlePerLowDataRequestsPersec", value)
}

// GetQuePerUsrIdlePerLowDataRequestsPersec gets the value of QuePerUsrIdlePerLowDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrIdlePerLowDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrIdlePerLowDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalAvgQueueLength sets the value of QuePerUsrNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrNormalAvgQueueLength", value)
}

// GetQuePerUsrNormalAvgQueueLength gets the value of QuePerUsrNormalAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalAvgQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalAvgsecPerDataRequest sets the value of QuePerUsrNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrNormalAvgsecPerDataRequest", value)
}

// GetQuePerUsrNormalAvgsecPerDataRequest gets the value of QuePerUsrNormalAvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalAvgsecPerDataRequest")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalAvgsecPerDataRequest_Base sets the value of QuePerUsrNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("QuePerUsrNormalAvgsecPerDataRequest_Base", value)
}

// GetQuePerUsrNormalAvgsecPerDataRequest_Base gets the value of QuePerUsrNormalAvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalAvgsecPerDataRequest_Base")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalBytesPersec sets the value of QuePerUsrNormalBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalBytesPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrNormalBytesPersec", value)
}

// GetQuePerUsrNormalBytesPersec gets the value of QuePerUsrNormalBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalBytesPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalCurrentQueueLength sets the value of QuePerUsrNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalCurrentQueueLength(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrNormalCurrentQueueLength", value)
}

// GetQuePerUsrNormalCurrentQueueLength gets the value of QuePerUsrNormalCurrentQueueLength for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalCurrentQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalCurrentQueueLength")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetQuePerUsrNormalDataRequestsPersec sets the value of QuePerUsrNormalDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) SetPropertyQuePerUsrNormalDataRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("QuePerUsrNormalDataRequestsPersec", value)
}

// GetQuePerUsrNormalDataRequestsPersec gets the value of QuePerUsrNormalDataRequestsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageDiskScheduler) GetPropertyQuePerUsrNormalDataRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("QuePerUsrNormalDataRequestsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}
