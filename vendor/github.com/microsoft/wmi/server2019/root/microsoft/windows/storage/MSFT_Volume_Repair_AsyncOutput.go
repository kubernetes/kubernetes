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

// MSFT_Volume_Repair_AsyncOutput struct
type MSFT_Volume_Repair_AsyncOutput struct {
	*MSFT_StorageJobOutParams

	//
	Output uint32
}

func NewMSFT_Volume_Repair_AsyncOutputEx1(instance *cim.WmiInstance) (newInstance *MSFT_Volume_Repair_AsyncOutput, err error) {
	tmp, err := NewMSFT_StorageJobOutParamsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_Volume_Repair_AsyncOutput{
		MSFT_StorageJobOutParams: tmp,
	}
	return
}

func NewMSFT_Volume_Repair_AsyncOutputEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_Volume_Repair_AsyncOutput, err error) {
	tmp, err := NewMSFT_StorageJobOutParamsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_Volume_Repair_AsyncOutput{
		MSFT_StorageJobOutParams: tmp,
	}
	return
}

// SetOutput sets the value of Output for the instance
func (instance *MSFT_Volume_Repair_AsyncOutput) SetPropertyOutput(value uint32) (err error) {
	return instance.SetProperty("Output", (value))
}

// GetOutput gets the value of Output for the instance
func (instance *MSFT_Volume_Repair_AsyncOutput) GetPropertyOutput() (value uint32, err error) {
	retValue, err := instance.GetProperty("Output")
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
