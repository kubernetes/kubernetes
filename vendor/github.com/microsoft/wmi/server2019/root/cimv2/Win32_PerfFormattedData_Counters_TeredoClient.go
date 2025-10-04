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

// Win32_PerfFormattedData_Counters_TeredoClient struct
type Win32_PerfFormattedData_Counters_TeredoClient struct {
	*Win32_PerfFormattedData

	//
	InTeredoBubble uint32

	//
	InTeredoData uint64

	//
	InTeredoDataKernelMode uint64

	//
	InTeredoDataUserMode uint64

	//
	InTeredoInvalid uint32

	//
	InTeredoRouterAdvertisement uint32

	//
	OutTeredoBubble uint32

	//
	OutTeredoData uint64

	//
	OutTeredoDataKernelMode uint64

	//
	OutTeredoDataUserMode uint64

	//
	OutTeredoRouterSolicitation uint32
}

func NewWin32_PerfFormattedData_Counters_TeredoClientEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_TeredoClient, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoClient{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_TeredoClientEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_TeredoClient, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoClient{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetInTeredoBubble sets the value of InTeredoBubble for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoBubble(value uint32) (err error) {
	return instance.SetProperty("InTeredoBubble", (value))
}

// GetInTeredoBubble gets the value of InTeredoBubble for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoBubble() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoBubble")
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

// SetInTeredoData sets the value of InTeredoData for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoData(value uint64) (err error) {
	return instance.SetProperty("InTeredoData", (value))
}

// GetInTeredoData gets the value of InTeredoData for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoData() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoData")
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

// SetInTeredoDataKernelMode sets the value of InTeredoDataKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoDataKernelMode(value uint64) (err error) {
	return instance.SetProperty("InTeredoDataKernelMode", (value))
}

// GetInTeredoDataKernelMode gets the value of InTeredoDataKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoDataKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoDataKernelMode")
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

// SetInTeredoDataUserMode sets the value of InTeredoDataUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoDataUserMode(value uint64) (err error) {
	return instance.SetProperty("InTeredoDataUserMode", (value))
}

// GetInTeredoDataUserMode gets the value of InTeredoDataUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoDataUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoDataUserMode")
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

// SetInTeredoInvalid sets the value of InTeredoInvalid for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoInvalid(value uint32) (err error) {
	return instance.SetProperty("InTeredoInvalid", (value))
}

// GetInTeredoInvalid gets the value of InTeredoInvalid for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoInvalid() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoInvalid")
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

// SetInTeredoRouterAdvertisement sets the value of InTeredoRouterAdvertisement for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyInTeredoRouterAdvertisement(value uint32) (err error) {
	return instance.SetProperty("InTeredoRouterAdvertisement", (value))
}

// GetInTeredoRouterAdvertisement gets the value of InTeredoRouterAdvertisement for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyInTeredoRouterAdvertisement() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRouterAdvertisement")
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

// SetOutTeredoBubble sets the value of OutTeredoBubble for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyOutTeredoBubble(value uint32) (err error) {
	return instance.SetProperty("OutTeredoBubble", (value))
}

// GetOutTeredoBubble gets the value of OutTeredoBubble for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyOutTeredoBubble() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoBubble")
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

// SetOutTeredoData sets the value of OutTeredoData for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyOutTeredoData(value uint64) (err error) {
	return instance.SetProperty("OutTeredoData", (value))
}

// GetOutTeredoData gets the value of OutTeredoData for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyOutTeredoData() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoData")
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

// SetOutTeredoDataKernelMode sets the value of OutTeredoDataKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyOutTeredoDataKernelMode(value uint64) (err error) {
	return instance.SetProperty("OutTeredoDataKernelMode", (value))
}

// GetOutTeredoDataKernelMode gets the value of OutTeredoDataKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyOutTeredoDataKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoDataKernelMode")
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

// SetOutTeredoDataUserMode sets the value of OutTeredoDataUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyOutTeredoDataUserMode(value uint64) (err error) {
	return instance.SetProperty("OutTeredoDataUserMode", (value))
}

// GetOutTeredoDataUserMode gets the value of OutTeredoDataUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyOutTeredoDataUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoDataUserMode")
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

// SetOutTeredoRouterSolicitation sets the value of OutTeredoRouterSolicitation for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) SetPropertyOutTeredoRouterSolicitation(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRouterSolicitation", (value))
}

// GetOutTeredoRouterSolicitation gets the value of OutTeredoRouterSolicitation for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoClient) GetPropertyOutTeredoRouterSolicitation() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRouterSolicitation")
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
