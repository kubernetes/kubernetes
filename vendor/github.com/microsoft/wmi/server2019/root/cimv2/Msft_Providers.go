// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Msft_Providers struct
type Msft_Providers struct {
	*cim.WmiInstance

	//
	HostingGroup string

	//
	HostingSpecification uint32

	//
	HostProcessIdentifier uint32

	//
	Locale string

	//
	Namespace string

	//
	provider string

	//
	ProviderOperation_AccessCheck uint64

	//
	ProviderOperation_CancelQuery uint64

	//
	ProviderOperation_CreateClassEnumAsync uint64

	//
	ProviderOperation_CreateInstanceEnumAsync uint64

	//
	ProviderOperation_CreateRefreshableEnum uint64

	//
	ProviderOperation_CreateRefreshableObject uint64

	//
	ProviderOperation_CreateRefresher uint64

	//
	ProviderOperation_DeleteClassAsync uint64

	//
	ProviderOperation_DeleteInstanceAsync uint64

	//
	ProviderOperation_ExecMethodAsync uint64

	//
	ProviderOperation_ExecQueryAsync uint64

	//
	ProviderOperation_FindConsumer uint64

	//
	ProviderOperation_GetObjectAsync uint64

	//
	ProviderOperation_GetObjects uint64

	//
	ProviderOperation_GetProperty uint64

	//
	ProviderOperation_NewQuery uint64

	//
	ProviderOperation_ProvideEvents uint64

	//
	ProviderOperation_PutClassAsync uint64

	//
	ProviderOperation_PutInstanceAsync uint64

	//
	ProviderOperation_PutProperty uint64

	//
	ProviderOperation_QueryInstances uint64

	//
	ProviderOperation_SetRegistrationObject uint64

	//
	ProviderOperation_StopRefreshing uint64

	//
	ProviderOperation_ValidateSubscription uint64

	//
	TransactionIdentifier string

	//
	User string
}

func NewMsft_ProvidersEx1(instance *cim.WmiInstance) (newInstance *Msft_Providers, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Msft_Providers{
		WmiInstance: tmp,
	}
	return
}

func NewMsft_ProvidersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_Providers, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_Providers{
		WmiInstance: tmp,
	}
	return
}

// SetHostingGroup sets the value of HostingGroup for the instance
func (instance *Msft_Providers) SetPropertyHostingGroup(value string) (err error) {
	return instance.SetProperty("HostingGroup", (value))
}

// GetHostingGroup gets the value of HostingGroup for the instance
func (instance *Msft_Providers) GetPropertyHostingGroup() (value string, err error) {
	retValue, err := instance.GetProperty("HostingGroup")
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

// SetHostingSpecification sets the value of HostingSpecification for the instance
func (instance *Msft_Providers) SetPropertyHostingSpecification(value uint32) (err error) {
	return instance.SetProperty("HostingSpecification", (value))
}

// GetHostingSpecification gets the value of HostingSpecification for the instance
func (instance *Msft_Providers) GetPropertyHostingSpecification() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostingSpecification")
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

// SetHostProcessIdentifier sets the value of HostProcessIdentifier for the instance
func (instance *Msft_Providers) SetPropertyHostProcessIdentifier(value uint32) (err error) {
	return instance.SetProperty("HostProcessIdentifier", (value))
}

// GetHostProcessIdentifier gets the value of HostProcessIdentifier for the instance
func (instance *Msft_Providers) GetPropertyHostProcessIdentifier() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostProcessIdentifier")
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

// SetLocale sets the value of Locale for the instance
func (instance *Msft_Providers) SetPropertyLocale(value string) (err error) {
	return instance.SetProperty("Locale", (value))
}

// GetLocale gets the value of Locale for the instance
func (instance *Msft_Providers) GetPropertyLocale() (value string, err error) {
	retValue, err := instance.GetProperty("Locale")
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

// SetNamespace sets the value of Namespace for the instance
func (instance *Msft_Providers) SetPropertyNamespace(value string) (err error) {
	return instance.SetProperty("Namespace", (value))
}

// GetNamespace gets the value of Namespace for the instance
func (instance *Msft_Providers) GetPropertyNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("Namespace")
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

// Setprovider sets the value of provider for the instance
func (instance *Msft_Providers) SetPropertyprovider(value string) (err error) {
	return instance.SetProperty("provider", (value))
}

// Getprovider gets the value of provider for the instance
func (instance *Msft_Providers) GetPropertyprovider() (value string, err error) {
	retValue, err := instance.GetProperty("provider")
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

// SetProviderOperation_AccessCheck sets the value of ProviderOperation_AccessCheck for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_AccessCheck(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_AccessCheck", (value))
}

// GetProviderOperation_AccessCheck gets the value of ProviderOperation_AccessCheck for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_AccessCheck() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_AccessCheck")
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

// SetProviderOperation_CancelQuery sets the value of ProviderOperation_CancelQuery for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CancelQuery(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CancelQuery", (value))
}

// GetProviderOperation_CancelQuery gets the value of ProviderOperation_CancelQuery for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CancelQuery() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CancelQuery")
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

// SetProviderOperation_CreateClassEnumAsync sets the value of ProviderOperation_CreateClassEnumAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CreateClassEnumAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CreateClassEnumAsync", (value))
}

// GetProviderOperation_CreateClassEnumAsync gets the value of ProviderOperation_CreateClassEnumAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CreateClassEnumAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CreateClassEnumAsync")
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

// SetProviderOperation_CreateInstanceEnumAsync sets the value of ProviderOperation_CreateInstanceEnumAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CreateInstanceEnumAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CreateInstanceEnumAsync", (value))
}

// GetProviderOperation_CreateInstanceEnumAsync gets the value of ProviderOperation_CreateInstanceEnumAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CreateInstanceEnumAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CreateInstanceEnumAsync")
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

// SetProviderOperation_CreateRefreshableEnum sets the value of ProviderOperation_CreateRefreshableEnum for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CreateRefreshableEnum(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CreateRefreshableEnum", (value))
}

// GetProviderOperation_CreateRefreshableEnum gets the value of ProviderOperation_CreateRefreshableEnum for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CreateRefreshableEnum() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CreateRefreshableEnum")
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

// SetProviderOperation_CreateRefreshableObject sets the value of ProviderOperation_CreateRefreshableObject for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CreateRefreshableObject(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CreateRefreshableObject", (value))
}

// GetProviderOperation_CreateRefreshableObject gets the value of ProviderOperation_CreateRefreshableObject for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CreateRefreshableObject() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CreateRefreshableObject")
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

// SetProviderOperation_CreateRefresher sets the value of ProviderOperation_CreateRefresher for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_CreateRefresher(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_CreateRefresher", (value))
}

// GetProviderOperation_CreateRefresher gets the value of ProviderOperation_CreateRefresher for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_CreateRefresher() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_CreateRefresher")
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

// SetProviderOperation_DeleteClassAsync sets the value of ProviderOperation_DeleteClassAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_DeleteClassAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_DeleteClassAsync", (value))
}

// GetProviderOperation_DeleteClassAsync gets the value of ProviderOperation_DeleteClassAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_DeleteClassAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_DeleteClassAsync")
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

// SetProviderOperation_DeleteInstanceAsync sets the value of ProviderOperation_DeleteInstanceAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_DeleteInstanceAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_DeleteInstanceAsync", (value))
}

// GetProviderOperation_DeleteInstanceAsync gets the value of ProviderOperation_DeleteInstanceAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_DeleteInstanceAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_DeleteInstanceAsync")
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

// SetProviderOperation_ExecMethodAsync sets the value of ProviderOperation_ExecMethodAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_ExecMethodAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_ExecMethodAsync", (value))
}

// GetProviderOperation_ExecMethodAsync gets the value of ProviderOperation_ExecMethodAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_ExecMethodAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_ExecMethodAsync")
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

// SetProviderOperation_ExecQueryAsync sets the value of ProviderOperation_ExecQueryAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_ExecQueryAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_ExecQueryAsync", (value))
}

// GetProviderOperation_ExecQueryAsync gets the value of ProviderOperation_ExecQueryAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_ExecQueryAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_ExecQueryAsync")
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

// SetProviderOperation_FindConsumer sets the value of ProviderOperation_FindConsumer for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_FindConsumer(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_FindConsumer", (value))
}

// GetProviderOperation_FindConsumer gets the value of ProviderOperation_FindConsumer for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_FindConsumer() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_FindConsumer")
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

// SetProviderOperation_GetObjectAsync sets the value of ProviderOperation_GetObjectAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_GetObjectAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_GetObjectAsync", (value))
}

// GetProviderOperation_GetObjectAsync gets the value of ProviderOperation_GetObjectAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_GetObjectAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_GetObjectAsync")
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

// SetProviderOperation_GetObjects sets the value of ProviderOperation_GetObjects for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_GetObjects(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_GetObjects", (value))
}

// GetProviderOperation_GetObjects gets the value of ProviderOperation_GetObjects for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_GetObjects() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_GetObjects")
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

// SetProviderOperation_GetProperty sets the value of ProviderOperation_GetProperty for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_GetProperty(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_GetProperty", (value))
}

// GetProviderOperation_GetProperty gets the value of ProviderOperation_GetProperty for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_GetProperty() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_GetProperty")
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

// SetProviderOperation_NewQuery sets the value of ProviderOperation_NewQuery for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_NewQuery(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_NewQuery", (value))
}

// GetProviderOperation_NewQuery gets the value of ProviderOperation_NewQuery for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_NewQuery() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_NewQuery")
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

// SetProviderOperation_ProvideEvents sets the value of ProviderOperation_ProvideEvents for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_ProvideEvents(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_ProvideEvents", (value))
}

// GetProviderOperation_ProvideEvents gets the value of ProviderOperation_ProvideEvents for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_ProvideEvents() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_ProvideEvents")
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

// SetProviderOperation_PutClassAsync sets the value of ProviderOperation_PutClassAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_PutClassAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_PutClassAsync", (value))
}

// GetProviderOperation_PutClassAsync gets the value of ProviderOperation_PutClassAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_PutClassAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_PutClassAsync")
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

// SetProviderOperation_PutInstanceAsync sets the value of ProviderOperation_PutInstanceAsync for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_PutInstanceAsync(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_PutInstanceAsync", (value))
}

// GetProviderOperation_PutInstanceAsync gets the value of ProviderOperation_PutInstanceAsync for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_PutInstanceAsync() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_PutInstanceAsync")
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

// SetProviderOperation_PutProperty sets the value of ProviderOperation_PutProperty for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_PutProperty(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_PutProperty", (value))
}

// GetProviderOperation_PutProperty gets the value of ProviderOperation_PutProperty for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_PutProperty() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_PutProperty")
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

// SetProviderOperation_QueryInstances sets the value of ProviderOperation_QueryInstances for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_QueryInstances(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_QueryInstances", (value))
}

// GetProviderOperation_QueryInstances gets the value of ProviderOperation_QueryInstances for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_QueryInstances() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_QueryInstances")
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

// SetProviderOperation_SetRegistrationObject sets the value of ProviderOperation_SetRegistrationObject for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_SetRegistrationObject(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_SetRegistrationObject", (value))
}

// GetProviderOperation_SetRegistrationObject gets the value of ProviderOperation_SetRegistrationObject for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_SetRegistrationObject() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_SetRegistrationObject")
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

// SetProviderOperation_StopRefreshing sets the value of ProviderOperation_StopRefreshing for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_StopRefreshing(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_StopRefreshing", (value))
}

// GetProviderOperation_StopRefreshing gets the value of ProviderOperation_StopRefreshing for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_StopRefreshing() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_StopRefreshing")
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

// SetProviderOperation_ValidateSubscription sets the value of ProviderOperation_ValidateSubscription for the instance
func (instance *Msft_Providers) SetPropertyProviderOperation_ValidateSubscription(value uint64) (err error) {
	return instance.SetProperty("ProviderOperation_ValidateSubscription", (value))
}

// GetProviderOperation_ValidateSubscription gets the value of ProviderOperation_ValidateSubscription for the instance
func (instance *Msft_Providers) GetPropertyProviderOperation_ValidateSubscription() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProviderOperation_ValidateSubscription")
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

// SetTransactionIdentifier sets the value of TransactionIdentifier for the instance
func (instance *Msft_Providers) SetPropertyTransactionIdentifier(value string) (err error) {
	return instance.SetProperty("TransactionIdentifier", (value))
}

// GetTransactionIdentifier gets the value of TransactionIdentifier for the instance
func (instance *Msft_Providers) GetPropertyTransactionIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("TransactionIdentifier")
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

// SetUser sets the value of User for the instance
func (instance *Msft_Providers) SetPropertyUser(value string) (err error) {
	return instance.SetProperty("User", (value))
}

// GetUser gets the value of User for the instance
func (instance *Msft_Providers) GetPropertyUser() (value string, err error) {
	retValue, err := instance.GetProperty("User")
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

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Msft_Providers) Suspend() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Suspend")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Msft_Providers) Resume() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Resume")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Msft_Providers) UnLoad() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("UnLoad")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Locale" type="string "></param>
// <param name="Namespace" type="string "></param>
// <param name="provider" type="string "></param>
// <param name="TransactionIdentifier" type="string "></param>
// <param name="User" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Msft_Providers) Load( /* IN */ Namespace string,
	/* IN */ User string,
	/* IN */ Locale string,
	/* IN */ provider string,
	/* IN */ TransactionIdentifier string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Load", Namespace, User, Locale, provider, TransactionIdentifier)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
