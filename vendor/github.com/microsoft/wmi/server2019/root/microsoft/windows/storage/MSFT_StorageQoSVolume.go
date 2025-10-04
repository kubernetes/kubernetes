// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageQoSVolume struct
type MSFT_StorageQoSVolume struct {
	*cim.WmiInstance

	//
	Bandwidth uint64

	//
	BandwidthLimit uint64

	//
	Interval uint64

	//
	IOPS uint64

	//
	Latency uint64

	//
	Limit uint64

	//
	Mountpoint string

	//
	Reservation uint64

	//
	Status uint16

	//
	TimeStamp uint64

	//
	VolumeId string
}

func NewMSFT_StorageQoSVolumeEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageQoSVolume, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSVolume{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageQoSVolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageQoSVolume, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSVolume{
		WmiInstance: tmp,
	}
	return
}

// SetBandwidth sets the value of Bandwidth for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyBandwidth(value uint64) (err error) {
	return instance.SetProperty("Bandwidth", (value))
}

// GetBandwidth gets the value of Bandwidth for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyBandwidth() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bandwidth")
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

// SetBandwidthLimit sets the value of BandwidthLimit for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyBandwidthLimit(value uint64) (err error) {
	return instance.SetProperty("BandwidthLimit", (value))
}

// GetBandwidthLimit gets the value of BandwidthLimit for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyBandwidthLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("BandwidthLimit")
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

// SetInterval sets the value of Interval for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyInterval(value uint64) (err error) {
	return instance.SetProperty("Interval", (value))
}

// GetInterval gets the value of Interval for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyInterval() (value uint64, err error) {
	retValue, err := instance.GetProperty("Interval")
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

// SetIOPS sets the value of IOPS for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyIOPS(value uint64) (err error) {
	return instance.SetProperty("IOPS", (value))
}

// GetIOPS gets the value of IOPS for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyIOPS() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOPS")
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

// SetLatency sets the value of Latency for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyLatency(value uint64) (err error) {
	return instance.SetProperty("Latency", (value))
}

// GetLatency gets the value of Latency for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("Latency")
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

// SetLimit sets the value of Limit for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyLimit(value uint64) (err error) {
	return instance.SetProperty("Limit", (value))
}

// GetLimit gets the value of Limit for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("Limit")
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

// SetMountpoint sets the value of Mountpoint for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyMountpoint(value string) (err error) {
	return instance.SetProperty("Mountpoint", (value))
}

// GetMountpoint gets the value of Mountpoint for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyMountpoint() (value string, err error) {
	retValue, err := instance.GetProperty("Mountpoint")
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

// SetReservation sets the value of Reservation for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyReservation(value uint64) (err error) {
	return instance.SetProperty("Reservation", (value))
}

// GetReservation gets the value of Reservation for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyReservation() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reservation")
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

// SetStatus sets the value of Status for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyStatus(value uint16) (err error) {
	return instance.SetProperty("Status", (value))
}

// GetStatus gets the value of Status for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("Status")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetTimeStamp sets the value of TimeStamp for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyTimeStamp(value uint64) (err error) {
	return instance.SetProperty("TimeStamp", (value))
}

// GetTimeStamp gets the value of TimeStamp for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyTimeStamp() (value uint64, err error) {
	retValue, err := instance.GetProperty("TimeStamp")
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

// SetVolumeId sets the value of VolumeId for the instance
func (instance *MSFT_StorageQoSVolume) SetPropertyVolumeId(value string) (err error) {
	return instance.SetProperty("VolumeId", (value))
}

// GetVolumeId gets the value of VolumeId for the instance
func (instance *MSFT_StorageQoSVolume) GetPropertyVolumeId() (value string, err error) {
	retValue, err := instance.GetProperty("VolumeId")
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
