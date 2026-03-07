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

// Win32_PerfRawData_Tcpip_IPv6 struct
type Win32_PerfRawData_Tcpip_IPv6 struct {
	*Win32_PerfRawData

	//
	DatagramsForwardedPersec uint32

	//
	DatagramsOutboundDiscarded uint32

	//
	DatagramsOutboundNoRoute uint32

	//
	DatagramsPersec uint32

	//
	DatagramsReceivedAddressErrors uint32

	//
	DatagramsReceivedDeliveredPersec uint32

	//
	DatagramsReceivedDiscarded uint32

	//
	DatagramsReceivedHeaderErrors uint32

	//
	DatagramsReceivedPersec uint32

	//
	DatagramsReceivedUnknownProtocol uint32

	//
	DatagramsSentPersec uint32

	//
	FragmentationFailures uint32

	//
	FragmentedDatagramsPersec uint32

	//
	FragmentReassemblyFailures uint32

	//
	FragmentsCreatedPersec uint32

	//
	FragmentsReassembledPersec uint32

	//
	FragmentsReceivedPersec uint32
}

func NewWin32_PerfRawData_Tcpip_IPv6Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Tcpip_IPv6, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_IPv6{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Tcpip_IPv6Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Tcpip_IPv6, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_IPv6{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDatagramsForwardedPersec sets the value of DatagramsForwardedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsForwardedPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsForwardedPersec", (value))
}

// GetDatagramsForwardedPersec gets the value of DatagramsForwardedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsForwardedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsForwardedPersec")
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

// SetDatagramsOutboundDiscarded sets the value of DatagramsOutboundDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsOutboundDiscarded(value uint32) (err error) {
	return instance.SetProperty("DatagramsOutboundDiscarded", (value))
}

// GetDatagramsOutboundDiscarded gets the value of DatagramsOutboundDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsOutboundDiscarded() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsOutboundDiscarded")
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

// SetDatagramsOutboundNoRoute sets the value of DatagramsOutboundNoRoute for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsOutboundNoRoute(value uint32) (err error) {
	return instance.SetProperty("DatagramsOutboundNoRoute", (value))
}

// GetDatagramsOutboundNoRoute gets the value of DatagramsOutboundNoRoute for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsOutboundNoRoute() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsOutboundNoRoute")
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

// SetDatagramsPersec sets the value of DatagramsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsPersec", (value))
}

// GetDatagramsPersec gets the value of DatagramsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsPersec")
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

// SetDatagramsReceivedAddressErrors sets the value of DatagramsReceivedAddressErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedAddressErrors(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedAddressErrors", (value))
}

// GetDatagramsReceivedAddressErrors gets the value of DatagramsReceivedAddressErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedAddressErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedAddressErrors")
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

// SetDatagramsReceivedDeliveredPersec sets the value of DatagramsReceivedDeliveredPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedDeliveredPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedDeliveredPersec", (value))
}

// GetDatagramsReceivedDeliveredPersec gets the value of DatagramsReceivedDeliveredPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedDeliveredPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedDeliveredPersec")
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

// SetDatagramsReceivedDiscarded sets the value of DatagramsReceivedDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedDiscarded(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedDiscarded", (value))
}

// GetDatagramsReceivedDiscarded gets the value of DatagramsReceivedDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedDiscarded() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedDiscarded")
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

// SetDatagramsReceivedHeaderErrors sets the value of DatagramsReceivedHeaderErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedHeaderErrors(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedHeaderErrors", (value))
}

// GetDatagramsReceivedHeaderErrors gets the value of DatagramsReceivedHeaderErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedHeaderErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedHeaderErrors")
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

// SetDatagramsReceivedPersec sets the value of DatagramsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedPersec", (value))
}

// GetDatagramsReceivedPersec gets the value of DatagramsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedPersec")
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

// SetDatagramsReceivedUnknownProtocol sets the value of DatagramsReceivedUnknownProtocol for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsReceivedUnknownProtocol(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedUnknownProtocol", (value))
}

// GetDatagramsReceivedUnknownProtocol gets the value of DatagramsReceivedUnknownProtocol for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsReceivedUnknownProtocol() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedUnknownProtocol")
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

// SetDatagramsSentPersec sets the value of DatagramsSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyDatagramsSentPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsSentPersec", (value))
}

// GetDatagramsSentPersec gets the value of DatagramsSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyDatagramsSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsSentPersec")
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

// SetFragmentationFailures sets the value of FragmentationFailures for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentationFailures(value uint32) (err error) {
	return instance.SetProperty("FragmentationFailures", (value))
}

// GetFragmentationFailures gets the value of FragmentationFailures for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentationFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentationFailures")
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

// SetFragmentedDatagramsPersec sets the value of FragmentedDatagramsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentedDatagramsPersec(value uint32) (err error) {
	return instance.SetProperty("FragmentedDatagramsPersec", (value))
}

// GetFragmentedDatagramsPersec gets the value of FragmentedDatagramsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentedDatagramsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentedDatagramsPersec")
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

// SetFragmentReassemblyFailures sets the value of FragmentReassemblyFailures for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentReassemblyFailures(value uint32) (err error) {
	return instance.SetProperty("FragmentReassemblyFailures", (value))
}

// GetFragmentReassemblyFailures gets the value of FragmentReassemblyFailures for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentReassemblyFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentReassemblyFailures")
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

// SetFragmentsCreatedPersec sets the value of FragmentsCreatedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentsCreatedPersec(value uint32) (err error) {
	return instance.SetProperty("FragmentsCreatedPersec", (value))
}

// GetFragmentsCreatedPersec gets the value of FragmentsCreatedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentsCreatedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentsCreatedPersec")
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

// SetFragmentsReassembledPersec sets the value of FragmentsReassembledPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentsReassembledPersec(value uint32) (err error) {
	return instance.SetProperty("FragmentsReassembledPersec", (value))
}

// GetFragmentsReassembledPersec gets the value of FragmentsReassembledPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentsReassembledPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentsReassembledPersec")
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

// SetFragmentsReceivedPersec sets the value of FragmentsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) SetPropertyFragmentsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("FragmentsReceivedPersec", (value))
}

// GetFragmentsReceivedPersec gets the value of FragmentsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_IPv6) GetPropertyFragmentsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FragmentsReceivedPersec")
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
