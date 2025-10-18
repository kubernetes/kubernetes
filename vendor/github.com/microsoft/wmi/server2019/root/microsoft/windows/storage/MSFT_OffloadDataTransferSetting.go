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

// MSFT_OffloadDataTransferSetting struct
type MSFT_OffloadDataTransferSetting struct {
	*MSFT_StorageObject

	//
	NumberOfTokensInUse uint32

	//
	NumberOfTokensMax uint32

	//
	OptimalDataTokenSize uint32

	//
	SupportInterSubsystem bool
}

func NewMSFT_OffloadDataTransferSettingEx1(instance *cim.WmiInstance) (newInstance *MSFT_OffloadDataTransferSetting, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_OffloadDataTransferSetting{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_OffloadDataTransferSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_OffloadDataTransferSetting, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_OffloadDataTransferSetting{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetNumberOfTokensInUse sets the value of NumberOfTokensInUse for the instance
func (instance *MSFT_OffloadDataTransferSetting) SetPropertyNumberOfTokensInUse(value uint32) (err error) {
	return instance.SetProperty("NumberOfTokensInUse", (value))
}

// GetNumberOfTokensInUse gets the value of NumberOfTokensInUse for the instance
func (instance *MSFT_OffloadDataTransferSetting) GetPropertyNumberOfTokensInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfTokensInUse")
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

// SetNumberOfTokensMax sets the value of NumberOfTokensMax for the instance
func (instance *MSFT_OffloadDataTransferSetting) SetPropertyNumberOfTokensMax(value uint32) (err error) {
	return instance.SetProperty("NumberOfTokensMax", (value))
}

// GetNumberOfTokensMax gets the value of NumberOfTokensMax for the instance
func (instance *MSFT_OffloadDataTransferSetting) GetPropertyNumberOfTokensMax() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfTokensMax")
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

// SetOptimalDataTokenSize sets the value of OptimalDataTokenSize for the instance
func (instance *MSFT_OffloadDataTransferSetting) SetPropertyOptimalDataTokenSize(value uint32) (err error) {
	return instance.SetProperty("OptimalDataTokenSize", (value))
}

// GetOptimalDataTokenSize gets the value of OptimalDataTokenSize for the instance
func (instance *MSFT_OffloadDataTransferSetting) GetPropertyOptimalDataTokenSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("OptimalDataTokenSize")
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

// SetSupportInterSubsystem sets the value of SupportInterSubsystem for the instance
func (instance *MSFT_OffloadDataTransferSetting) SetPropertySupportInterSubsystem(value bool) (err error) {
	return instance.SetProperty("SupportInterSubsystem", (value))
}

// GetSupportInterSubsystem gets the value of SupportInterSubsystem for the instance
func (instance *MSFT_OffloadDataTransferSetting) GetPropertySupportInterSubsystem() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportInterSubsystem")
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
