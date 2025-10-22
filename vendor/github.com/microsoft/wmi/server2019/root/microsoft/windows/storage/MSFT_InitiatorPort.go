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

// MSFT_InitiatorPort struct
type MSFT_InitiatorPort struct {
	*cim.WmiInstance

	//
	AlternateNodeAddress []string

	//
	AlternatePortAddress []string

	//
	ConnectionType uint16

	//
	InstanceName string

	//
	NodeAddress string

	//
	ObjectId string

	//
	OperationalStatus []uint16

	//
	OtherConnectionTypeDescription string

	//
	PortAddress string

	//
	PortType uint16
}

func NewMSFT_InitiatorPortEx1(instance *cim.WmiInstance) (newInstance *MSFT_InitiatorPort, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPort{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_InitiatorPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_InitiatorPort, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPort{
		WmiInstance: tmp,
	}
	return
}

// SetAlternateNodeAddress sets the value of AlternateNodeAddress for the instance
func (instance *MSFT_InitiatorPort) SetPropertyAlternateNodeAddress(value []string) (err error) {
	return instance.SetProperty("AlternateNodeAddress", (value))
}

// GetAlternateNodeAddress gets the value of AlternateNodeAddress for the instance
func (instance *MSFT_InitiatorPort) GetPropertyAlternateNodeAddress() (value []string, err error) {
	retValue, err := instance.GetProperty("AlternateNodeAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetAlternatePortAddress sets the value of AlternatePortAddress for the instance
func (instance *MSFT_InitiatorPort) SetPropertyAlternatePortAddress(value []string) (err error) {
	return instance.SetProperty("AlternatePortAddress", (value))
}

// GetAlternatePortAddress gets the value of AlternatePortAddress for the instance
func (instance *MSFT_InitiatorPort) GetPropertyAlternatePortAddress() (value []string, err error) {
	retValue, err := instance.GetProperty("AlternatePortAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetConnectionType sets the value of ConnectionType for the instance
func (instance *MSFT_InitiatorPort) SetPropertyConnectionType(value uint16) (err error) {
	return instance.SetProperty("ConnectionType", (value))
}

// GetConnectionType gets the value of ConnectionType for the instance
func (instance *MSFT_InitiatorPort) GetPropertyConnectionType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ConnectionType")
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

// SetInstanceName sets the value of InstanceName for the instance
func (instance *MSFT_InitiatorPort) SetPropertyInstanceName(value string) (err error) {
	return instance.SetProperty("InstanceName", (value))
}

// GetInstanceName gets the value of InstanceName for the instance
func (instance *MSFT_InitiatorPort) GetPropertyInstanceName() (value string, err error) {
	retValue, err := instance.GetProperty("InstanceName")
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

// SetNodeAddress sets the value of NodeAddress for the instance
func (instance *MSFT_InitiatorPort) SetPropertyNodeAddress(value string) (err error) {
	return instance.SetProperty("NodeAddress", (value))
}

// GetNodeAddress gets the value of NodeAddress for the instance
func (instance *MSFT_InitiatorPort) GetPropertyNodeAddress() (value string, err error) {
	retValue, err := instance.GetProperty("NodeAddress")
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

// SetObjectId sets the value of ObjectId for the instance
func (instance *MSFT_InitiatorPort) SetPropertyObjectId(value string) (err error) {
	return instance.SetProperty("ObjectId", (value))
}

// GetObjectId gets the value of ObjectId for the instance
func (instance *MSFT_InitiatorPort) GetPropertyObjectId() (value string, err error) {
	retValue, err := instance.GetProperty("ObjectId")
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_InitiatorPort) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_InitiatorPort) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetOtherConnectionTypeDescription sets the value of OtherConnectionTypeDescription for the instance
func (instance *MSFT_InitiatorPort) SetPropertyOtherConnectionTypeDescription(value string) (err error) {
	return instance.SetProperty("OtherConnectionTypeDescription", (value))
}

// GetOtherConnectionTypeDescription gets the value of OtherConnectionTypeDescription for the instance
func (instance *MSFT_InitiatorPort) GetPropertyOtherConnectionTypeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherConnectionTypeDescription")
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

// SetPortAddress sets the value of PortAddress for the instance
func (instance *MSFT_InitiatorPort) SetPropertyPortAddress(value string) (err error) {
	return instance.SetProperty("PortAddress", (value))
}

// GetPortAddress gets the value of PortAddress for the instance
func (instance *MSFT_InitiatorPort) GetPropertyPortAddress() (value string, err error) {
	retValue, err := instance.GetProperty("PortAddress")
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

// SetPortType sets the value of PortType for the instance
func (instance *MSFT_InitiatorPort) SetPropertyPortType(value uint16) (err error) {
	return instance.SetProperty("PortType", (value))
}

// GetPortType gets the value of PortType for the instance
func (instance *MSFT_InitiatorPort) GetPropertyPortType() (value uint16, err error) {
	retValue, err := instance.GetProperty("PortType")
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

//

// <param name="NodeAddress" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_InitiatorPort) SetNodeAddress( /* IN */ NodeAddress string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetNodeAddress", NodeAddress)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
