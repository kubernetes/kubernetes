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

// MSFT_ResiliencySetting struct
type MSFT_ResiliencySetting struct {
	*MSFT_StorageObject

	//
	Description string

	//
	InterleaveDefault uint64

	//
	InterleaveMax uint64

	//
	InterleaveMin uint64

	//
	Name string

	//
	NumberOfColumnsDefault uint16

	//
	NumberOfColumnsMax uint16

	//
	NumberOfColumnsMin uint16

	//
	NumberOfDataCopiesDefault uint16

	//
	NumberOfDataCopiesMax uint16

	//
	NumberOfDataCopiesMin uint16

	//
	NumberOfGroupsDefault uint16

	//
	NumberOfGroupsMax uint16

	//
	NumberOfGroupsMin uint16

	//
	ParityLayout uint16

	//
	PhysicalDiskRedundancyDefault uint16

	//
	PhysicalDiskRedundancyMax uint16

	//
	PhysicalDiskRedundancyMin uint16

	//
	RequestNoSinglePointOfFailure bool
}

func NewMSFT_ResiliencySettingEx1(instance *cim.WmiInstance) (newInstance *MSFT_ResiliencySetting, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_ResiliencySetting{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_ResiliencySettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_ResiliencySetting, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_ResiliencySetting{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetDescription sets the value of Description for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetInterleaveDefault sets the value of InterleaveDefault for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyInterleaveDefault(value uint64) (err error) {
	return instance.SetProperty("InterleaveDefault", (value))
}

// GetInterleaveDefault gets the value of InterleaveDefault for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyInterleaveDefault() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterleaveDefault")
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

// SetInterleaveMax sets the value of InterleaveMax for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyInterleaveMax(value uint64) (err error) {
	return instance.SetProperty("InterleaveMax", (value))
}

// GetInterleaveMax gets the value of InterleaveMax for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyInterleaveMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterleaveMax")
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

// SetInterleaveMin sets the value of InterleaveMin for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyInterleaveMin(value uint64) (err error) {
	return instance.SetProperty("InterleaveMin", (value))
}

// GetInterleaveMin gets the value of InterleaveMin for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyInterleaveMin() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterleaveMin")
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
func (instance *MSFT_ResiliencySetting) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyName() (value string, err error) {
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

// SetNumberOfColumnsDefault sets the value of NumberOfColumnsDefault for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfColumnsDefault(value uint16) (err error) {
	return instance.SetProperty("NumberOfColumnsDefault", (value))
}

// GetNumberOfColumnsDefault gets the value of NumberOfColumnsDefault for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfColumnsDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfColumnsDefault")
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

// SetNumberOfColumnsMax sets the value of NumberOfColumnsMax for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfColumnsMax(value uint16) (err error) {
	return instance.SetProperty("NumberOfColumnsMax", (value))
}

// GetNumberOfColumnsMax gets the value of NumberOfColumnsMax for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfColumnsMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfColumnsMax")
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

// SetNumberOfColumnsMin sets the value of NumberOfColumnsMin for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfColumnsMin(value uint16) (err error) {
	return instance.SetProperty("NumberOfColumnsMin", (value))
}

// GetNumberOfColumnsMin gets the value of NumberOfColumnsMin for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfColumnsMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfColumnsMin")
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

// SetNumberOfDataCopiesDefault sets the value of NumberOfDataCopiesDefault for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfDataCopiesDefault(value uint16) (err error) {
	return instance.SetProperty("NumberOfDataCopiesDefault", (value))
}

// GetNumberOfDataCopiesDefault gets the value of NumberOfDataCopiesDefault for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfDataCopiesDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfDataCopiesDefault")
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

// SetNumberOfDataCopiesMax sets the value of NumberOfDataCopiesMax for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfDataCopiesMax(value uint16) (err error) {
	return instance.SetProperty("NumberOfDataCopiesMax", (value))
}

// GetNumberOfDataCopiesMax gets the value of NumberOfDataCopiesMax for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfDataCopiesMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfDataCopiesMax")
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

// SetNumberOfDataCopiesMin sets the value of NumberOfDataCopiesMin for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfDataCopiesMin(value uint16) (err error) {
	return instance.SetProperty("NumberOfDataCopiesMin", (value))
}

// GetNumberOfDataCopiesMin gets the value of NumberOfDataCopiesMin for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfDataCopiesMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfDataCopiesMin")
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

// SetNumberOfGroupsDefault sets the value of NumberOfGroupsDefault for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfGroupsDefault(value uint16) (err error) {
	return instance.SetProperty("NumberOfGroupsDefault", (value))
}

// GetNumberOfGroupsDefault gets the value of NumberOfGroupsDefault for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfGroupsDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfGroupsDefault")
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

// SetNumberOfGroupsMax sets the value of NumberOfGroupsMax for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfGroupsMax(value uint16) (err error) {
	return instance.SetProperty("NumberOfGroupsMax", (value))
}

// GetNumberOfGroupsMax gets the value of NumberOfGroupsMax for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfGroupsMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfGroupsMax")
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

// SetNumberOfGroupsMin sets the value of NumberOfGroupsMin for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyNumberOfGroupsMin(value uint16) (err error) {
	return instance.SetProperty("NumberOfGroupsMin", (value))
}

// GetNumberOfGroupsMin gets the value of NumberOfGroupsMin for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyNumberOfGroupsMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfGroupsMin")
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

// SetParityLayout sets the value of ParityLayout for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyParityLayout(value uint16) (err error) {
	return instance.SetProperty("ParityLayout", (value))
}

// GetParityLayout gets the value of ParityLayout for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyParityLayout() (value uint16, err error) {
	retValue, err := instance.GetProperty("ParityLayout")
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

// SetPhysicalDiskRedundancyDefault sets the value of PhysicalDiskRedundancyDefault for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyPhysicalDiskRedundancyDefault(value uint16) (err error) {
	return instance.SetProperty("PhysicalDiskRedundancyDefault", (value))
}

// GetPhysicalDiskRedundancyDefault gets the value of PhysicalDiskRedundancyDefault for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyPhysicalDiskRedundancyDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskRedundancyDefault")
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

// SetPhysicalDiskRedundancyMax sets the value of PhysicalDiskRedundancyMax for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyPhysicalDiskRedundancyMax(value uint16) (err error) {
	return instance.SetProperty("PhysicalDiskRedundancyMax", (value))
}

// GetPhysicalDiskRedundancyMax gets the value of PhysicalDiskRedundancyMax for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyPhysicalDiskRedundancyMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskRedundancyMax")
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

// SetPhysicalDiskRedundancyMin sets the value of PhysicalDiskRedundancyMin for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyPhysicalDiskRedundancyMin(value uint16) (err error) {
	return instance.SetProperty("PhysicalDiskRedundancyMin", (value))
}

// GetPhysicalDiskRedundancyMin gets the value of PhysicalDiskRedundancyMin for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyPhysicalDiskRedundancyMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskRedundancyMin")
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

// SetRequestNoSinglePointOfFailure sets the value of RequestNoSinglePointOfFailure for the instance
func (instance *MSFT_ResiliencySetting) SetPropertyRequestNoSinglePointOfFailure(value bool) (err error) {
	return instance.SetProperty("RequestNoSinglePointOfFailure", (value))
}

// GetRequestNoSinglePointOfFailure gets the value of RequestNoSinglePointOfFailure for the instance
func (instance *MSFT_ResiliencySetting) GetPropertyRequestNoSinglePointOfFailure() (value bool, err error) {
	retValue, err := instance.GetProperty("RequestNoSinglePointOfFailure")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

//

// <param name="AutoNumberOfColumns" type="bool "></param>
// <param name="InterleaveDefault" type="uint64 "></param>
// <param name="NumberOfColumnsDefault" type="uint16 "></param>
// <param name="NumberOfDataCopiesDefault" type="uint16 "></param>
// <param name="NumberOfGroupsDefault" type="uint16 "></param>
// <param name="PhysicalDiskRedundancyDefault" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_ResiliencySetting) SetDefaults( /* IN */ NumberOfDataCopiesDefault uint16,
	/* IN */ PhysicalDiskRedundancyDefault uint16,
	/* IN */ NumberOfColumnsDefault uint16,
	/* IN */ AutoNumberOfColumns bool,
	/* IN */ InterleaveDefault uint64,
	/* IN */ NumberOfGroupsDefault uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetDefaults", NumberOfDataCopiesDefault, PhysicalDiskRedundancyDefault, NumberOfColumnsDefault, AutoNumberOfColumns, InterleaveDefault, NumberOfGroupsDefault)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
