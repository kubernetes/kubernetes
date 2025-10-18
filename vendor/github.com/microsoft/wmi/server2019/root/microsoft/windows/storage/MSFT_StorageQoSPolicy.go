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

// MSFT_StorageQoSPolicy struct
type MSFT_StorageQoSPolicy struct {
	*cim.WmiInstance

	//
	BandwidthLimit uint64

	//
	Name string

	//
	ParentPolicy string

	//
	PolicyId string

	//
	PolicyType uint16

	//
	Status uint16

	//
	ThroughputLimit uint64

	//
	ThroughputReservation uint64
}

func NewMSFT_StorageQoSPolicyEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageQoSPolicy, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicy{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageQoSPolicyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageQoSPolicy, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicy{
		WmiInstance: tmp,
	}
	return
}

// SetBandwidthLimit sets the value of BandwidthLimit for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyBandwidthLimit(value uint64) (err error) {
	return instance.SetProperty("BandwidthLimit", (value))
}

// GetBandwidthLimit gets the value of BandwidthLimit for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyBandwidthLimit() (value uint64, err error) {
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

// SetName sets the value of Name for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetParentPolicy sets the value of ParentPolicy for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyParentPolicy(value string) (err error) {
	return instance.SetProperty("ParentPolicy", (value))
}

// GetParentPolicy gets the value of ParentPolicy for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyParentPolicy() (value string, err error) {
	retValue, err := instance.GetProperty("ParentPolicy")
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

// SetPolicyId sets the value of PolicyId for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyPolicyId(value string) (err error) {
	return instance.SetProperty("PolicyId", (value))
}

// GetPolicyId gets the value of PolicyId for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyPolicyId() (value string, err error) {
	retValue, err := instance.GetProperty("PolicyId")
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

// SetPolicyType sets the value of PolicyType for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyPolicyType(value uint16) (err error) {
	return instance.SetProperty("PolicyType", (value))
}

// GetPolicyType gets the value of PolicyType for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyPolicyType() (value uint16, err error) {
	retValue, err := instance.GetProperty("PolicyType")
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

// SetStatus sets the value of Status for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyStatus(value uint16) (err error) {
	return instance.SetProperty("Status", (value))
}

// GetStatus gets the value of Status for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyStatus() (value uint16, err error) {
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

// SetThroughputLimit sets the value of ThroughputLimit for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyThroughputLimit(value uint64) (err error) {
	return instance.SetProperty("ThroughputLimit", (value))
}

// GetThroughputLimit gets the value of ThroughputLimit for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyThroughputLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThroughputLimit")
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

// SetThroughputReservation sets the value of ThroughputReservation for the instance
func (instance *MSFT_StorageQoSPolicy) SetPropertyThroughputReservation(value uint64) (err error) {
	return instance.SetProperty("ThroughputReservation", (value))
}

// GetThroughputReservation gets the value of ThroughputReservation for the instance
func (instance *MSFT_StorageQoSPolicy) GetPropertyThroughputReservation() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThroughputReservation")
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

//

// <param name="BandwidthLimit" type="uint64 "></param>
// <param name="Limit" type="uint64 "></param>
// <param name="NewName" type="string "></param>
// <param name="Reservation" type="uint64 "></param>

// <param name="ReturnValue" type="int32 "></param>
func (instance *MSFT_StorageQoSPolicy) SetAttributes( /* IN */ NewName string,
	/* IN */ Limit uint64,
	/* IN */ Reservation uint64,
	/* IN */ BandwidthLimit uint64) (result int32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetAttributes", NewName, Limit, Reservation, BandwidthLimit)
	if err != nil {
		return
	}
	result = int32(retVal)
	return

}

//

// <param name="ReturnValue" type="int32 "></param>
func (instance *MSFT_StorageQoSPolicy) DeletePolicy() (result int32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DeletePolicy")
	if err != nil {
		return
	}
	result = int32(retVal)
	return

}
