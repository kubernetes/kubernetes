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

// Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice struct
type Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice struct {
	*Win32_PerfFormattedData

	//
	NumberofconnectedVMTchannels uint32

	//
	NumberofcreatedVMTchannels uint32

	//
	NumberofdisconnectedVMTchannels uint32

	//
	NumberofRDVGMrestartednotifications uint32

	//
	NumberofwaitingVMTchannels uint32

	//
	TotalnumberofcreatedVMTchannels uint32
}

func NewWin32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetNumberofconnectedVMTchannels sets the value of NumberofconnectedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyNumberofconnectedVMTchannels(value uint32) (err error) {
	return instance.SetProperty("NumberofconnectedVMTchannels", (value))
}

// GetNumberofconnectedVMTchannels gets the value of NumberofconnectedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyNumberofconnectedVMTchannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofconnectedVMTchannels")
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

// SetNumberofcreatedVMTchannels sets the value of NumberofcreatedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyNumberofcreatedVMTchannels(value uint32) (err error) {
	return instance.SetProperty("NumberofcreatedVMTchannels", (value))
}

// GetNumberofcreatedVMTchannels gets the value of NumberofcreatedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyNumberofcreatedVMTchannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofcreatedVMTchannels")
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

// SetNumberofdisconnectedVMTchannels sets the value of NumberofdisconnectedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyNumberofdisconnectedVMTchannels(value uint32) (err error) {
	return instance.SetProperty("NumberofdisconnectedVMTchannels", (value))
}

// GetNumberofdisconnectedVMTchannels gets the value of NumberofdisconnectedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyNumberofdisconnectedVMTchannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofdisconnectedVMTchannels")
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

// SetNumberofRDVGMrestartednotifications sets the value of NumberofRDVGMrestartednotifications for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyNumberofRDVGMrestartednotifications(value uint32) (err error) {
	return instance.SetProperty("NumberofRDVGMrestartednotifications", (value))
}

// GetNumberofRDVGMrestartednotifications gets the value of NumberofRDVGMrestartednotifications for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyNumberofRDVGMrestartednotifications() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofRDVGMrestartednotifications")
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

// SetNumberofwaitingVMTchannels sets the value of NumberofwaitingVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyNumberofwaitingVMTchannels(value uint32) (err error) {
	return instance.SetProperty("NumberofwaitingVMTchannels", (value))
}

// GetNumberofwaitingVMTchannels gets the value of NumberofwaitingVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyNumberofwaitingVMTchannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofwaitingVMTchannels")
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

// SetTotalnumberofcreatedVMTchannels sets the value of TotalnumberofcreatedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) SetPropertyTotalnumberofcreatedVMTchannels(value uint32) (err error) {
	return instance.SetProperty("TotalnumberofcreatedVMTchannels", (value))
}

// GetTotalnumberofcreatedVMTchannels gets the value of TotalnumberofcreatedVMTchannels for the instance
func (instance *Win32_PerfFormattedData_MicrosoftWindowsRemoteDesktopServicesRemoteFXSynth3dvsc_RemoteFXSynth3DVSCVMDevice) GetPropertyTotalnumberofcreatedVMTchannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalnumberofcreatedVMTchannels")
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
