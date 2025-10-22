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

// MSFT_DedupProperties struct
type MSFT_DedupProperties struct {
	*cim.WmiInstance

	//
	InPolicyFilesCount uint64

	//
	InPolicyFilesSize uint64

	//
	OptimizedFilesCount uint64

	//
	OptimizedFilesSavingsRate uint32

	//
	OptimizedFilesSize uint64

	//
	SavingsRate uint32

	//
	SavingsSize uint64

	//
	UnoptimizedSize uint64
}

func NewMSFT_DedupPropertiesEx1(instance *cim.WmiInstance) (newInstance *MSFT_DedupProperties, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_DedupProperties{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_DedupPropertiesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_DedupProperties, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_DedupProperties{
		WmiInstance: tmp,
	}
	return
}

// SetInPolicyFilesCount sets the value of InPolicyFilesCount for the instance
func (instance *MSFT_DedupProperties) SetPropertyInPolicyFilesCount(value uint64) (err error) {
	return instance.SetProperty("InPolicyFilesCount", (value))
}

// GetInPolicyFilesCount gets the value of InPolicyFilesCount for the instance
func (instance *MSFT_DedupProperties) GetPropertyInPolicyFilesCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("InPolicyFilesCount")
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

// SetInPolicyFilesSize sets the value of InPolicyFilesSize for the instance
func (instance *MSFT_DedupProperties) SetPropertyInPolicyFilesSize(value uint64) (err error) {
	return instance.SetProperty("InPolicyFilesSize", (value))
}

// GetInPolicyFilesSize gets the value of InPolicyFilesSize for the instance
func (instance *MSFT_DedupProperties) GetPropertyInPolicyFilesSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("InPolicyFilesSize")
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

// SetOptimizedFilesCount sets the value of OptimizedFilesCount for the instance
func (instance *MSFT_DedupProperties) SetPropertyOptimizedFilesCount(value uint64) (err error) {
	return instance.SetProperty("OptimizedFilesCount", (value))
}

// GetOptimizedFilesCount gets the value of OptimizedFilesCount for the instance
func (instance *MSFT_DedupProperties) GetPropertyOptimizedFilesCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("OptimizedFilesCount")
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

// SetOptimizedFilesSavingsRate sets the value of OptimizedFilesSavingsRate for the instance
func (instance *MSFT_DedupProperties) SetPropertyOptimizedFilesSavingsRate(value uint32) (err error) {
	return instance.SetProperty("OptimizedFilesSavingsRate", (value))
}

// GetOptimizedFilesSavingsRate gets the value of OptimizedFilesSavingsRate for the instance
func (instance *MSFT_DedupProperties) GetPropertyOptimizedFilesSavingsRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("OptimizedFilesSavingsRate")
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

// SetOptimizedFilesSize sets the value of OptimizedFilesSize for the instance
func (instance *MSFT_DedupProperties) SetPropertyOptimizedFilesSize(value uint64) (err error) {
	return instance.SetProperty("OptimizedFilesSize", (value))
}

// GetOptimizedFilesSize gets the value of OptimizedFilesSize for the instance
func (instance *MSFT_DedupProperties) GetPropertyOptimizedFilesSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("OptimizedFilesSize")
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

// SetSavingsRate sets the value of SavingsRate for the instance
func (instance *MSFT_DedupProperties) SetPropertySavingsRate(value uint32) (err error) {
	return instance.SetProperty("SavingsRate", (value))
}

// GetSavingsRate gets the value of SavingsRate for the instance
func (instance *MSFT_DedupProperties) GetPropertySavingsRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("SavingsRate")
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

// SetSavingsSize sets the value of SavingsSize for the instance
func (instance *MSFT_DedupProperties) SetPropertySavingsSize(value uint64) (err error) {
	return instance.SetProperty("SavingsSize", (value))
}

// GetSavingsSize gets the value of SavingsSize for the instance
func (instance *MSFT_DedupProperties) GetPropertySavingsSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("SavingsSize")
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

// SetUnoptimizedSize sets the value of UnoptimizedSize for the instance
func (instance *MSFT_DedupProperties) SetPropertyUnoptimizedSize(value uint64) (err error) {
	return instance.SetProperty("UnoptimizedSize", (value))
}

// GetUnoptimizedSize gets the value of UnoptimizedSize for the instance
func (instance *MSFT_DedupProperties) GetPropertyUnoptimizedSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnoptimizedSize")
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
