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

// __PropertyProviderRegistration struct
type __PropertyProviderRegistration struct {
	*__ProviderRegistration

	//
	SupportsGet bool

	//
	SupportsPut bool
}

func New__PropertyProviderRegistrationEx1(instance *cim.WmiInstance) (newInstance *__PropertyProviderRegistration, err error) {
	tmp, err := New__ProviderRegistrationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__PropertyProviderRegistration{
		__ProviderRegistration: tmp,
	}
	return
}

func New__PropertyProviderRegistrationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__PropertyProviderRegistration, err error) {
	tmp, err := New__ProviderRegistrationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__PropertyProviderRegistration{
		__ProviderRegistration: tmp,
	}
	return
}

// SetSupportsGet sets the value of SupportsGet for the instance
func (instance *__PropertyProviderRegistration) SetPropertySupportsGet(value bool) (err error) {
	return instance.SetProperty("SupportsGet", (value))
}

// GetSupportsGet gets the value of SupportsGet for the instance
func (instance *__PropertyProviderRegistration) GetPropertySupportsGet() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsGet")
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

// SetSupportsPut sets the value of SupportsPut for the instance
func (instance *__PropertyProviderRegistration) SetPropertySupportsPut(value bool) (err error) {
	return instance.SetProperty("SupportsPut", (value))
}

// GetSupportsPut gets the value of SupportsPut for the instance
func (instance *__PropertyProviderRegistration) GetPropertySupportsPut() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsPut")
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
