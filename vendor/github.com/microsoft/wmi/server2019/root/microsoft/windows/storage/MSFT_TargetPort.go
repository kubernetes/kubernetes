// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_TargetPort struct
type MSFT_TargetPort struct {
	*MSFT_StorageObject

	//
	ConnectionType uint16

	//
	FriendlyName string

	//
	HealthStatus uint16

	//
	LinkTechnology uint16

	//
	MaxSpeed uint64

	//
	NetworkAddresses []string

	//
	NodeAddress string

	//
	OperationalStatus []uint16

	//
	OtherConnectionTypeDescription string

	//
	OtherLinkTechnology string

	//
	OtherOperationalStatusDescription string

	//
	PortAddress string

	//
	PortNumbers []uint16

	//
	PortType uint16

	//
	Role uint16

	//
	Speed uint64

	//
	StorageControllerId string

	//
	UsageRestriction uint16
}

func NewMSFT_TargetPortEx1(instance *cim.WmiInstance) (newInstance *MSFT_TargetPort, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_TargetPort{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_TargetPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_TargetPort, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_TargetPort{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetConnectionType sets the value of ConnectionType for the instance
func (instance *MSFT_TargetPort) SetPropertyConnectionType(value uint16) (err error) {
	return instance.SetProperty("ConnectionType", (value))
}

// GetConnectionType gets the value of ConnectionType for the instance
func (instance *MSFT_TargetPort) GetPropertyConnectionType() (value uint16, err error) {
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

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *MSFT_TargetPort) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_TargetPort) GetPropertyFriendlyName() (value string, err error) {
	retValue, err := instance.GetProperty("FriendlyName")
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_TargetPort) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_TargetPort) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
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

// SetLinkTechnology sets the value of LinkTechnology for the instance
func (instance *MSFT_TargetPort) SetPropertyLinkTechnology(value uint16) (err error) {
	return instance.SetProperty("LinkTechnology", (value))
}

// GetLinkTechnology gets the value of LinkTechnology for the instance
func (instance *MSFT_TargetPort) GetPropertyLinkTechnology() (value uint16, err error) {
	retValue, err := instance.GetProperty("LinkTechnology")
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

// SetMaxSpeed sets the value of MaxSpeed for the instance
func (instance *MSFT_TargetPort) SetPropertyMaxSpeed(value uint64) (err error) {
	return instance.SetProperty("MaxSpeed", (value))
}

// GetMaxSpeed gets the value of MaxSpeed for the instance
func (instance *MSFT_TargetPort) GetPropertyMaxSpeed() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxSpeed")
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

// SetNetworkAddresses sets the value of NetworkAddresses for the instance
func (instance *MSFT_TargetPort) SetPropertyNetworkAddresses(value []string) (err error) {
	return instance.SetProperty("NetworkAddresses", (value))
}

// GetNetworkAddresses gets the value of NetworkAddresses for the instance
func (instance *MSFT_TargetPort) GetPropertyNetworkAddresses() (value []string, err error) {
	retValue, err := instance.GetProperty("NetworkAddresses")
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

// SetNodeAddress sets the value of NodeAddress for the instance
func (instance *MSFT_TargetPort) SetPropertyNodeAddress(value string) (err error) {
	return instance.SetProperty("NodeAddress", (value))
}

// GetNodeAddress gets the value of NodeAddress for the instance
func (instance *MSFT_TargetPort) GetPropertyNodeAddress() (value string, err error) {
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_TargetPort) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_TargetPort) GetPropertyOperationalStatus() (value []uint16, err error) {
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
func (instance *MSFT_TargetPort) SetPropertyOtherConnectionTypeDescription(value string) (err error) {
	return instance.SetProperty("OtherConnectionTypeDescription", (value))
}

// GetOtherConnectionTypeDescription gets the value of OtherConnectionTypeDescription for the instance
func (instance *MSFT_TargetPort) GetPropertyOtherConnectionTypeDescription() (value string, err error) {
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

// SetOtherLinkTechnology sets the value of OtherLinkTechnology for the instance
func (instance *MSFT_TargetPort) SetPropertyOtherLinkTechnology(value string) (err error) {
	return instance.SetProperty("OtherLinkTechnology", (value))
}

// GetOtherLinkTechnology gets the value of OtherLinkTechnology for the instance
func (instance *MSFT_TargetPort) GetPropertyOtherLinkTechnology() (value string, err error) {
	retValue, err := instance.GetProperty("OtherLinkTechnology")
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

// SetOtherOperationalStatusDescription sets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_TargetPort) SetPropertyOtherOperationalStatusDescription(value string) (err error) {
	return instance.SetProperty("OtherOperationalStatusDescription", (value))
}

// GetOtherOperationalStatusDescription gets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_TargetPort) GetPropertyOtherOperationalStatusDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherOperationalStatusDescription")
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
func (instance *MSFT_TargetPort) SetPropertyPortAddress(value string) (err error) {
	return instance.SetProperty("PortAddress", (value))
}

// GetPortAddress gets the value of PortAddress for the instance
func (instance *MSFT_TargetPort) GetPropertyPortAddress() (value string, err error) {
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

// SetPortNumbers sets the value of PortNumbers for the instance
func (instance *MSFT_TargetPort) SetPropertyPortNumbers(value []uint16) (err error) {
	return instance.SetProperty("PortNumbers", (value))
}

// GetPortNumbers gets the value of PortNumbers for the instance
func (instance *MSFT_TargetPort) GetPropertyPortNumbers() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PortNumbers")
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

// SetPortType sets the value of PortType for the instance
func (instance *MSFT_TargetPort) SetPropertyPortType(value uint16) (err error) {
	return instance.SetProperty("PortType", (value))
}

// GetPortType gets the value of PortType for the instance
func (instance *MSFT_TargetPort) GetPropertyPortType() (value uint16, err error) {
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

// SetRole sets the value of Role for the instance
func (instance *MSFT_TargetPort) SetPropertyRole(value uint16) (err error) {
	return instance.SetProperty("Role", (value))
}

// GetRole gets the value of Role for the instance
func (instance *MSFT_TargetPort) GetPropertyRole() (value uint16, err error) {
	retValue, err := instance.GetProperty("Role")
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

// SetSpeed sets the value of Speed for the instance
func (instance *MSFT_TargetPort) SetPropertySpeed(value uint64) (err error) {
	return instance.SetProperty("Speed", (value))
}

// GetSpeed gets the value of Speed for the instance
func (instance *MSFT_TargetPort) GetPropertySpeed() (value uint64, err error) {
	retValue, err := instance.GetProperty("Speed")
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

// SetStorageControllerId sets the value of StorageControllerId for the instance
func (instance *MSFT_TargetPort) SetPropertyStorageControllerId(value string) (err error) {
	return instance.SetProperty("StorageControllerId", (value))
}

// GetStorageControllerId gets the value of StorageControllerId for the instance
func (instance *MSFT_TargetPort) GetPropertyStorageControllerId() (value string, err error) {
	retValue, err := instance.GetProperty("StorageControllerId")
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

// SetUsageRestriction sets the value of UsageRestriction for the instance
func (instance *MSFT_TargetPort) SetPropertyUsageRestriction(value uint16) (err error) {
	return instance.SetProperty("UsageRestriction", (value))
}

// GetUsageRestriction gets the value of UsageRestriction for the instance
func (instance *MSFT_TargetPort) GetPropertyUsageRestriction() (value uint16, err error) {
	retValue, err := instance.GetProperty("UsageRestriction")
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
