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

// MSFT_StorageFaultDomainToStorageFaultDomain struct
type MSFT_StorageFaultDomainToStorageFaultDomain struct {
	*cim.WmiInstance

	//
	SourceStorageFaultDomain MSFT_StorageFaultDomain

	//
	TargetStorageFaultDomain MSFT_StorageFaultDomain
}

func NewMSFT_StorageFaultDomainToStorageFaultDomainEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageFaultDomainToStorageFaultDomain, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageFaultDomainToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageFaultDomainToStorageFaultDomainEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageFaultDomainToStorageFaultDomain, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageFaultDomainToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

// SetSourceStorageFaultDomain sets the value of SourceStorageFaultDomain for the instance
func (instance *MSFT_StorageFaultDomainToStorageFaultDomain) SetPropertySourceStorageFaultDomain(value MSFT_StorageFaultDomain) (err error) {
	return instance.SetProperty("SourceStorageFaultDomain", (value))
}

// GetSourceStorageFaultDomain gets the value of SourceStorageFaultDomain for the instance
func (instance *MSFT_StorageFaultDomainToStorageFaultDomain) GetPropertySourceStorageFaultDomain() (value MSFT_StorageFaultDomain, err error) {
	retValue, err := instance.GetProperty("SourceStorageFaultDomain")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageFaultDomain)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageFaultDomain is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageFaultDomain(valuetmp)

	return
}

// SetTargetStorageFaultDomain sets the value of TargetStorageFaultDomain for the instance
func (instance *MSFT_StorageFaultDomainToStorageFaultDomain) SetPropertyTargetStorageFaultDomain(value MSFT_StorageFaultDomain) (err error) {
	return instance.SetProperty("TargetStorageFaultDomain", (value))
}

// GetTargetStorageFaultDomain gets the value of TargetStorageFaultDomain for the instance
func (instance *MSFT_StorageFaultDomainToStorageFaultDomain) GetPropertyTargetStorageFaultDomain() (value MSFT_StorageFaultDomain, err error) {
	retValue, err := instance.GetProperty("TargetStorageFaultDomain")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageFaultDomain)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageFaultDomain is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageFaultDomain(valuetmp)

	return
}
