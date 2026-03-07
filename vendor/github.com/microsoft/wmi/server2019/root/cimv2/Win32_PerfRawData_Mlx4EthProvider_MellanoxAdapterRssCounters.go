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

// Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters struct
type Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters struct {
	*Win32_PerfRawData

	//
	EncapsulatedNonRssIPv4Only uint64

	//
	EncapsulatedNonRssIPv4PerTcp uint32

	//
	EncapsulatedNonRssIPv4PerUdp uint32

	//
	EncapsulatedNonRssIPv6Only uint64

	//
	EncapsulatedNonRssIPv6PerTcp uint32

	//
	EncapsulatedNonRssIPv6PerUdp uint32

	//
	EncapsulatedNonRssMisc uint32

	//
	EncapsulatedRssIPv4Only uint64

	//
	EncapsulatedRssIPv4PerTcp uint32

	//
	EncapsulatedRssIPv4PerUdp uint32

	//
	EncapsulatedRssIPv6Only uint64

	//
	EncapsulatedRssIPv6PerTcp uint32

	//
	EncapsulatedRssIPv6PerUdp uint32

	//
	EncapsulatedRssMisc uint32

	//
	NonRssIPv4Only uint64

	//
	NonRssIPv4PerTcp uint32

	//
	NonRssIPv4PerUdp uint32

	//
	NonRssIPv6Only uint64

	//
	NonRssIPv6PerTcp uint32

	//
	NonRssIPv6PerUdp uint32

	//
	NonRssMisc uint32

	//
	RssIPv4Only uint64

	//
	RssIPv4PerTcp uint32

	//
	RssIPv4PerUdp uint32

	//
	RssIPv6Only uint64

	//
	RssIPv6PerTcp uint32

	//
	RssIPv6PerUdp uint32

	//
	RssMisc uint32
}

func NewWin32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetEncapsulatedNonRssIPv4Only sets the value of EncapsulatedNonRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv4Only(value uint64) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv4Only", (value))
}

// GetEncapsulatedNonRssIPv4Only gets the value of EncapsulatedNonRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv4Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv4Only")
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

// SetEncapsulatedNonRssIPv4PerTcp sets the value of EncapsulatedNonRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv4PerTcp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv4PerTcp", (value))
}

// GetEncapsulatedNonRssIPv4PerTcp gets the value of EncapsulatedNonRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv4PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv4PerTcp")
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

// SetEncapsulatedNonRssIPv4PerUdp sets the value of EncapsulatedNonRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv4PerUdp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv4PerUdp", (value))
}

// GetEncapsulatedNonRssIPv4PerUdp gets the value of EncapsulatedNonRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv4PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv4PerUdp")
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

// SetEncapsulatedNonRssIPv6Only sets the value of EncapsulatedNonRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv6Only(value uint64) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv6Only", (value))
}

// GetEncapsulatedNonRssIPv6Only gets the value of EncapsulatedNonRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv6Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv6Only")
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

// SetEncapsulatedNonRssIPv6PerTcp sets the value of EncapsulatedNonRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv6PerTcp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv6PerTcp", (value))
}

// GetEncapsulatedNonRssIPv6PerTcp gets the value of EncapsulatedNonRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv6PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv6PerTcp")
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

// SetEncapsulatedNonRssIPv6PerUdp sets the value of EncapsulatedNonRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssIPv6PerUdp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedNonRssIPv6PerUdp", (value))
}

// GetEncapsulatedNonRssIPv6PerUdp gets the value of EncapsulatedNonRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssIPv6PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssIPv6PerUdp")
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

// SetEncapsulatedNonRssMisc sets the value of EncapsulatedNonRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedNonRssMisc(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedNonRssMisc", (value))
}

// GetEncapsulatedNonRssMisc gets the value of EncapsulatedNonRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedNonRssMisc() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedNonRssMisc")
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

// SetEncapsulatedRssIPv4Only sets the value of EncapsulatedRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv4Only(value uint64) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv4Only", (value))
}

// GetEncapsulatedRssIPv4Only gets the value of EncapsulatedRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv4Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv4Only")
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

// SetEncapsulatedRssIPv4PerTcp sets the value of EncapsulatedRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv4PerTcp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv4PerTcp", (value))
}

// GetEncapsulatedRssIPv4PerTcp gets the value of EncapsulatedRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv4PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv4PerTcp")
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

// SetEncapsulatedRssIPv4PerUdp sets the value of EncapsulatedRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv4PerUdp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv4PerUdp", (value))
}

// GetEncapsulatedRssIPv4PerUdp gets the value of EncapsulatedRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv4PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv4PerUdp")
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

// SetEncapsulatedRssIPv6Only sets the value of EncapsulatedRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv6Only(value uint64) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv6Only", (value))
}

// GetEncapsulatedRssIPv6Only gets the value of EncapsulatedRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv6Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv6Only")
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

// SetEncapsulatedRssIPv6PerTcp sets the value of EncapsulatedRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv6PerTcp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv6PerTcp", (value))
}

// GetEncapsulatedRssIPv6PerTcp gets the value of EncapsulatedRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv6PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv6PerTcp")
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

// SetEncapsulatedRssIPv6PerUdp sets the value of EncapsulatedRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssIPv6PerUdp(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedRssIPv6PerUdp", (value))
}

// GetEncapsulatedRssIPv6PerUdp gets the value of EncapsulatedRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssIPv6PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssIPv6PerUdp")
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

// SetEncapsulatedRssMisc sets the value of EncapsulatedRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyEncapsulatedRssMisc(value uint32) (err error) {
	return instance.SetProperty("EncapsulatedRssMisc", (value))
}

// GetEncapsulatedRssMisc gets the value of EncapsulatedRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyEncapsulatedRssMisc() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncapsulatedRssMisc")
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

// SetNonRssIPv4Only sets the value of NonRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv4Only(value uint64) (err error) {
	return instance.SetProperty("NonRssIPv4Only", (value))
}

// GetNonRssIPv4Only gets the value of NonRssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv4Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonRssIPv4Only")
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

// SetNonRssIPv4PerTcp sets the value of NonRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv4PerTcp(value uint32) (err error) {
	return instance.SetProperty("NonRssIPv4PerTcp", (value))
}

// GetNonRssIPv4PerTcp gets the value of NonRssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv4PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonRssIPv4PerTcp")
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

// SetNonRssIPv4PerUdp sets the value of NonRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv4PerUdp(value uint32) (err error) {
	return instance.SetProperty("NonRssIPv4PerUdp", (value))
}

// GetNonRssIPv4PerUdp gets the value of NonRssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv4PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonRssIPv4PerUdp")
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

// SetNonRssIPv6Only sets the value of NonRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv6Only(value uint64) (err error) {
	return instance.SetProperty("NonRssIPv6Only", (value))
}

// GetNonRssIPv6Only gets the value of NonRssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv6Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonRssIPv6Only")
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

// SetNonRssIPv6PerTcp sets the value of NonRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv6PerTcp(value uint32) (err error) {
	return instance.SetProperty("NonRssIPv6PerTcp", (value))
}

// GetNonRssIPv6PerTcp gets the value of NonRssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv6PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonRssIPv6PerTcp")
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

// SetNonRssIPv6PerUdp sets the value of NonRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssIPv6PerUdp(value uint32) (err error) {
	return instance.SetProperty("NonRssIPv6PerUdp", (value))
}

// GetNonRssIPv6PerUdp gets the value of NonRssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssIPv6PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonRssIPv6PerUdp")
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

// SetNonRssMisc sets the value of NonRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyNonRssMisc(value uint32) (err error) {
	return instance.SetProperty("NonRssMisc", (value))
}

// GetNonRssMisc gets the value of NonRssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyNonRssMisc() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonRssMisc")
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

// SetRssIPv4Only sets the value of RssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv4Only(value uint64) (err error) {
	return instance.SetProperty("RssIPv4Only", (value))
}

// GetRssIPv4Only gets the value of RssIPv4Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv4Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("RssIPv4Only")
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

// SetRssIPv4PerTcp sets the value of RssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv4PerTcp(value uint32) (err error) {
	return instance.SetProperty("RssIPv4PerTcp", (value))
}

// GetRssIPv4PerTcp gets the value of RssIPv4PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv4PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("RssIPv4PerTcp")
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

// SetRssIPv4PerUdp sets the value of RssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv4PerUdp(value uint32) (err error) {
	return instance.SetProperty("RssIPv4PerUdp", (value))
}

// GetRssIPv4PerUdp gets the value of RssIPv4PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv4PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("RssIPv4PerUdp")
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

// SetRssIPv6Only sets the value of RssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv6Only(value uint64) (err error) {
	return instance.SetProperty("RssIPv6Only", (value))
}

// GetRssIPv6Only gets the value of RssIPv6Only for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv6Only() (value uint64, err error) {
	retValue, err := instance.GetProperty("RssIPv6Only")
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

// SetRssIPv6PerTcp sets the value of RssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv6PerTcp(value uint32) (err error) {
	return instance.SetProperty("RssIPv6PerTcp", (value))
}

// GetRssIPv6PerTcp gets the value of RssIPv6PerTcp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv6PerTcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("RssIPv6PerTcp")
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

// SetRssIPv6PerUdp sets the value of RssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssIPv6PerUdp(value uint32) (err error) {
	return instance.SetProperty("RssIPv6PerUdp", (value))
}

// GetRssIPv6PerUdp gets the value of RssIPv6PerUdp for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssIPv6PerUdp() (value uint32, err error) {
	retValue, err := instance.GetProperty("RssIPv6PerUdp")
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

// SetRssMisc sets the value of RssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) SetPropertyRssMisc(value uint32) (err error) {
	return instance.SetProperty("RssMisc", (value))
}

// GetRssMisc gets the value of RssMisc for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterRssCounters) GetPropertyRssMisc() (value uint32, err error) {
	retValue, err := instance.GetProperty("RssMisc")
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
