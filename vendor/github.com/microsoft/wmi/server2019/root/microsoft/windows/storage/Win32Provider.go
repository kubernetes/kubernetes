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

// __Win32Provider struct
type __Win32Provider struct {
	*__Provider

	//
	ClientLoadableCLSID string

	//
	CLSID string

	//
	Concurrency int32

	//
	DefaultMachineName string

	//
	Enabled bool

	//
	HostingModel string

	//
	ImpersonationLevel Win32Provider_ImpersonationLevel

	//
	InitializationReentrancy Win32Provider_InitializationReentrancy

	//
	InitializationTimeoutInterval string

	//
	InitializeAsAdminFirst bool

	//
	OperationTimeoutInterval string

	//
	PerLocaleInitialization bool

	//
	PerUserInitialization bool

	//
	Pure bool

	//
	SecurityDescriptor string

	//
	SupportsExplicitShutdown bool

	//
	SupportsExtendedStatus bool

	//
	SupportsQuotas bool

	//
	SupportsSendStatus bool

	//
	SupportsShutdown bool

	//
	SupportsThrottling bool

	//
	UnloadTimeout string

	//
	Version uint32
}

func New__Win32ProviderEx1(instance *cim.WmiInstance) (newInstance *__Win32Provider, err error) {
	tmp, err := New__ProviderEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__Win32Provider{
		__Provider: tmp,
	}
	return
}

func New__Win32ProviderEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__Win32Provider, err error) {
	tmp, err := New__ProviderEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__Win32Provider{
		__Provider: tmp,
	}
	return
}

// SetClientLoadableCLSID sets the value of ClientLoadableCLSID for the instance
func (instance *__Win32Provider) SetPropertyClientLoadableCLSID(value string) (err error) {
	return instance.SetProperty("ClientLoadableCLSID", (value))
}

// GetClientLoadableCLSID gets the value of ClientLoadableCLSID for the instance
func (instance *__Win32Provider) GetPropertyClientLoadableCLSID() (value string, err error) {
	retValue, err := instance.GetProperty("ClientLoadableCLSID")
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

// SetCLSID sets the value of CLSID for the instance
func (instance *__Win32Provider) SetPropertyCLSID(value string) (err error) {
	return instance.SetProperty("CLSID", (value))
}

// GetCLSID gets the value of CLSID for the instance
func (instance *__Win32Provider) GetPropertyCLSID() (value string, err error) {
	retValue, err := instance.GetProperty("CLSID")
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

// SetConcurrency sets the value of Concurrency for the instance
func (instance *__Win32Provider) SetPropertyConcurrency(value int32) (err error) {
	return instance.SetProperty("Concurrency", (value))
}

// GetConcurrency gets the value of Concurrency for the instance
func (instance *__Win32Provider) GetPropertyConcurrency() (value int32, err error) {
	retValue, err := instance.GetProperty("Concurrency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}

// SetDefaultMachineName sets the value of DefaultMachineName for the instance
func (instance *__Win32Provider) SetPropertyDefaultMachineName(value string) (err error) {
	return instance.SetProperty("DefaultMachineName", (value))
}

// GetDefaultMachineName gets the value of DefaultMachineName for the instance
func (instance *__Win32Provider) GetPropertyDefaultMachineName() (value string, err error) {
	retValue, err := instance.GetProperty("DefaultMachineName")
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

// SetEnabled sets the value of Enabled for the instance
func (instance *__Win32Provider) SetPropertyEnabled(value bool) (err error) {
	return instance.SetProperty("Enabled", (value))
}

// GetEnabled gets the value of Enabled for the instance
func (instance *__Win32Provider) GetPropertyEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("Enabled")
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

// SetHostingModel sets the value of HostingModel for the instance
func (instance *__Win32Provider) SetPropertyHostingModel(value string) (err error) {
	return instance.SetProperty("HostingModel", (value))
}

// GetHostingModel gets the value of HostingModel for the instance
func (instance *__Win32Provider) GetPropertyHostingModel() (value string, err error) {
	retValue, err := instance.GetProperty("HostingModel")
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

// SetImpersonationLevel sets the value of ImpersonationLevel for the instance
func (instance *__Win32Provider) SetPropertyImpersonationLevel(value Win32Provider_ImpersonationLevel) (err error) {
	return instance.SetProperty("ImpersonationLevel", (value))
}

// GetImpersonationLevel gets the value of ImpersonationLevel for the instance
func (instance *__Win32Provider) GetPropertyImpersonationLevel() (value Win32Provider_ImpersonationLevel, err error) {
	retValue, err := instance.GetProperty("ImpersonationLevel")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32Provider_ImpersonationLevel(valuetmp)

	return
}

// SetInitializationReentrancy sets the value of InitializationReentrancy for the instance
func (instance *__Win32Provider) SetPropertyInitializationReentrancy(value Win32Provider_InitializationReentrancy) (err error) {
	return instance.SetProperty("InitializationReentrancy", (value))
}

// GetInitializationReentrancy gets the value of InitializationReentrancy for the instance
func (instance *__Win32Provider) GetPropertyInitializationReentrancy() (value Win32Provider_InitializationReentrancy, err error) {
	retValue, err := instance.GetProperty("InitializationReentrancy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32Provider_InitializationReentrancy(valuetmp)

	return
}

// SetInitializationTimeoutInterval sets the value of InitializationTimeoutInterval for the instance
func (instance *__Win32Provider) SetPropertyInitializationTimeoutInterval(value string) (err error) {
	return instance.SetProperty("InitializationTimeoutInterval", (value))
}

// GetInitializationTimeoutInterval gets the value of InitializationTimeoutInterval for the instance
func (instance *__Win32Provider) GetPropertyInitializationTimeoutInterval() (value string, err error) {
	retValue, err := instance.GetProperty("InitializationTimeoutInterval")
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

// SetInitializeAsAdminFirst sets the value of InitializeAsAdminFirst for the instance
func (instance *__Win32Provider) SetPropertyInitializeAsAdminFirst(value bool) (err error) {
	return instance.SetProperty("InitializeAsAdminFirst", (value))
}

// GetInitializeAsAdminFirst gets the value of InitializeAsAdminFirst for the instance
func (instance *__Win32Provider) GetPropertyInitializeAsAdminFirst() (value bool, err error) {
	retValue, err := instance.GetProperty("InitializeAsAdminFirst")
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

// SetOperationTimeoutInterval sets the value of OperationTimeoutInterval for the instance
func (instance *__Win32Provider) SetPropertyOperationTimeoutInterval(value string) (err error) {
	return instance.SetProperty("OperationTimeoutInterval", (value))
}

// GetOperationTimeoutInterval gets the value of OperationTimeoutInterval for the instance
func (instance *__Win32Provider) GetPropertyOperationTimeoutInterval() (value string, err error) {
	retValue, err := instance.GetProperty("OperationTimeoutInterval")
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

// SetPerLocaleInitialization sets the value of PerLocaleInitialization for the instance
func (instance *__Win32Provider) SetPropertyPerLocaleInitialization(value bool) (err error) {
	return instance.SetProperty("PerLocaleInitialization", (value))
}

// GetPerLocaleInitialization gets the value of PerLocaleInitialization for the instance
func (instance *__Win32Provider) GetPropertyPerLocaleInitialization() (value bool, err error) {
	retValue, err := instance.GetProperty("PerLocaleInitialization")
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

// SetPerUserInitialization sets the value of PerUserInitialization for the instance
func (instance *__Win32Provider) SetPropertyPerUserInitialization(value bool) (err error) {
	return instance.SetProperty("PerUserInitialization", (value))
}

// GetPerUserInitialization gets the value of PerUserInitialization for the instance
func (instance *__Win32Provider) GetPropertyPerUserInitialization() (value bool, err error) {
	retValue, err := instance.GetProperty("PerUserInitialization")
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

// SetPure sets the value of Pure for the instance
func (instance *__Win32Provider) SetPropertyPure(value bool) (err error) {
	return instance.SetProperty("Pure", (value))
}

// GetPure gets the value of Pure for the instance
func (instance *__Win32Provider) GetPropertyPure() (value bool, err error) {
	retValue, err := instance.GetProperty("Pure")
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

// SetSecurityDescriptor sets the value of SecurityDescriptor for the instance
func (instance *__Win32Provider) SetPropertySecurityDescriptor(value string) (err error) {
	return instance.SetProperty("SecurityDescriptor", (value))
}

// GetSecurityDescriptor gets the value of SecurityDescriptor for the instance
func (instance *__Win32Provider) GetPropertySecurityDescriptor() (value string, err error) {
	retValue, err := instance.GetProperty("SecurityDescriptor")
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

// SetSupportsExplicitShutdown sets the value of SupportsExplicitShutdown for the instance
func (instance *__Win32Provider) SetPropertySupportsExplicitShutdown(value bool) (err error) {
	return instance.SetProperty("SupportsExplicitShutdown", (value))
}

// GetSupportsExplicitShutdown gets the value of SupportsExplicitShutdown for the instance
func (instance *__Win32Provider) GetPropertySupportsExplicitShutdown() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsExplicitShutdown")
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

// SetSupportsExtendedStatus sets the value of SupportsExtendedStatus for the instance
func (instance *__Win32Provider) SetPropertySupportsExtendedStatus(value bool) (err error) {
	return instance.SetProperty("SupportsExtendedStatus", (value))
}

// GetSupportsExtendedStatus gets the value of SupportsExtendedStatus for the instance
func (instance *__Win32Provider) GetPropertySupportsExtendedStatus() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsExtendedStatus")
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

// SetSupportsQuotas sets the value of SupportsQuotas for the instance
func (instance *__Win32Provider) SetPropertySupportsQuotas(value bool) (err error) {
	return instance.SetProperty("SupportsQuotas", (value))
}

// GetSupportsQuotas gets the value of SupportsQuotas for the instance
func (instance *__Win32Provider) GetPropertySupportsQuotas() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsQuotas")
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

// SetSupportsSendStatus sets the value of SupportsSendStatus for the instance
func (instance *__Win32Provider) SetPropertySupportsSendStatus(value bool) (err error) {
	return instance.SetProperty("SupportsSendStatus", (value))
}

// GetSupportsSendStatus gets the value of SupportsSendStatus for the instance
func (instance *__Win32Provider) GetPropertySupportsSendStatus() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsSendStatus")
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

// SetSupportsShutdown sets the value of SupportsShutdown for the instance
func (instance *__Win32Provider) SetPropertySupportsShutdown(value bool) (err error) {
	return instance.SetProperty("SupportsShutdown", (value))
}

// GetSupportsShutdown gets the value of SupportsShutdown for the instance
func (instance *__Win32Provider) GetPropertySupportsShutdown() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsShutdown")
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

// SetSupportsThrottling sets the value of SupportsThrottling for the instance
func (instance *__Win32Provider) SetPropertySupportsThrottling(value bool) (err error) {
	return instance.SetProperty("SupportsThrottling", (value))
}

// GetSupportsThrottling gets the value of SupportsThrottling for the instance
func (instance *__Win32Provider) GetPropertySupportsThrottling() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsThrottling")
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

// SetUnloadTimeout sets the value of UnloadTimeout for the instance
func (instance *__Win32Provider) SetPropertyUnloadTimeout(value string) (err error) {
	return instance.SetProperty("UnloadTimeout", (value))
}

// GetUnloadTimeout gets the value of UnloadTimeout for the instance
func (instance *__Win32Provider) GetPropertyUnloadTimeout() (value string, err error) {
	retValue, err := instance.GetProperty("UnloadTimeout")
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

// SetVersion sets the value of Version for the instance
func (instance *__Win32Provider) SetPropertyVersion(value uint32) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *__Win32Provider) GetPropertyVersion() (value uint32, err error) {
	retValue, err := instance.GetProperty("Version")
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
