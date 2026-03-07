// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_PerfRawData_Lsa_SecuritySystemWideStatistics struct
type Win32_PerfRawData_Lsa_SecuritySystemWideStatistics struct {
	*Win32_PerfRawData

	//
	ActiveSchannelSessionCacheEntries uint32

	//
	DigestAuthentications uint32

	//
	ForwardedKerberosRequests uint32

	//
	KDCarmoredASRequests uint32

	//
	KDCarmoredTGSRequests uint32

	//
	KDCASRequests uint32

	//
	KDCclaimsawareASRequests uint32

	//
	KDCclaimsawareserviceassertedidentityTGSrequests uint32

	//
	KDCclaimsawareTGSRequests uint32

	//
	KDCclassictypeconstraineddelegationTGSRequests uint32

	//
	KDCkeytrustASRequests uint32

	//
	KDCresourcetypeconstraineddelegationTGSRequests uint32

	//
	KDCTGSRequests uint32

	//
	KerberosAuthentications uint32

	//
	NTLMAuthentications uint32

	//
	SchannelSessionCacheEntries uint32

	//
	SSLClientSideFullHandshakes uint32

	//
	SSLClientSideReconnectHandshakes uint32

	//
	SSLServerSideFullHandshakes uint32

	//
	SSLServerSideReconnectHandshakes uint32
}

func NewWin32_PerfRawData_Lsa_SecuritySystemWideStatisticsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Lsa_SecuritySystemWideStatistics{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Lsa_SecuritySystemWideStatisticsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Lsa_SecuritySystemWideStatistics{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetActiveSchannelSessionCacheEntries sets the value of ActiveSchannelSessionCacheEntries for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyActiveSchannelSessionCacheEntries(value uint32) (err error) {
	return instance.SetProperty("ActiveSchannelSessionCacheEntries", (value))
}

// GetActiveSchannelSessionCacheEntries gets the value of ActiveSchannelSessionCacheEntries for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyActiveSchannelSessionCacheEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveSchannelSessionCacheEntries")
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

// SetDigestAuthentications sets the value of DigestAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyDigestAuthentications(value uint32) (err error) {
	return instance.SetProperty("DigestAuthentications", (value))
}

// GetDigestAuthentications gets the value of DigestAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyDigestAuthentications() (value uint32, err error) {
	retValue, err := instance.GetProperty("DigestAuthentications")
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

// SetForwardedKerberosRequests sets the value of ForwardedKerberosRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyForwardedKerberosRequests(value uint32) (err error) {
	return instance.SetProperty("ForwardedKerberosRequests", (value))
}

// GetForwardedKerberosRequests gets the value of ForwardedKerberosRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyForwardedKerberosRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("ForwardedKerberosRequests")
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

// SetKDCarmoredASRequests sets the value of KDCarmoredASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCarmoredASRequests(value uint32) (err error) {
	return instance.SetProperty("KDCarmoredASRequests", (value))
}

// GetKDCarmoredASRequests gets the value of KDCarmoredASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCarmoredASRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCarmoredASRequests")
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

// SetKDCarmoredTGSRequests sets the value of KDCarmoredTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCarmoredTGSRequests(value uint32) (err error) {
	return instance.SetProperty("KDCarmoredTGSRequests", (value))
}

// GetKDCarmoredTGSRequests gets the value of KDCarmoredTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCarmoredTGSRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCarmoredTGSRequests")
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

// SetKDCASRequests sets the value of KDCASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCASRequests(value uint32) (err error) {
	return instance.SetProperty("KDCASRequests", (value))
}

// GetKDCASRequests gets the value of KDCASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCASRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCASRequests")
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

// SetKDCclaimsawareASRequests sets the value of KDCclaimsawareASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCclaimsawareASRequests(value uint32) (err error) {
	return instance.SetProperty("KDCclaimsawareASRequests", (value))
}

// GetKDCclaimsawareASRequests gets the value of KDCclaimsawareASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCclaimsawareASRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCclaimsawareASRequests")
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

// SetKDCclaimsawareserviceassertedidentityTGSrequests sets the value of KDCclaimsawareserviceassertedidentityTGSrequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCclaimsawareserviceassertedidentityTGSrequests(value uint32) (err error) {
	return instance.SetProperty("KDCclaimsawareserviceassertedidentityTGSrequests", (value))
}

// GetKDCclaimsawareserviceassertedidentityTGSrequests gets the value of KDCclaimsawareserviceassertedidentityTGSrequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCclaimsawareserviceassertedidentityTGSrequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCclaimsawareserviceassertedidentityTGSrequests")
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

// SetKDCclaimsawareTGSRequests sets the value of KDCclaimsawareTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCclaimsawareTGSRequests(value uint32) (err error) {
	return instance.SetProperty("KDCclaimsawareTGSRequests", (value))
}

// GetKDCclaimsawareTGSRequests gets the value of KDCclaimsawareTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCclaimsawareTGSRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCclaimsawareTGSRequests")
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

// SetKDCclassictypeconstraineddelegationTGSRequests sets the value of KDCclassictypeconstraineddelegationTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCclassictypeconstraineddelegationTGSRequests(value uint32) (err error) {
	return instance.SetProperty("KDCclassictypeconstraineddelegationTGSRequests", (value))
}

// GetKDCclassictypeconstraineddelegationTGSRequests gets the value of KDCclassictypeconstraineddelegationTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCclassictypeconstraineddelegationTGSRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCclassictypeconstraineddelegationTGSRequests")
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

// SetKDCkeytrustASRequests sets the value of KDCkeytrustASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCkeytrustASRequests(value uint32) (err error) {
	return instance.SetProperty("KDCkeytrustASRequests", (value))
}

// GetKDCkeytrustASRequests gets the value of KDCkeytrustASRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCkeytrustASRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCkeytrustASRequests")
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

// SetKDCresourcetypeconstraineddelegationTGSRequests sets the value of KDCresourcetypeconstraineddelegationTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCresourcetypeconstraineddelegationTGSRequests(value uint32) (err error) {
	return instance.SetProperty("KDCresourcetypeconstraineddelegationTGSRequests", (value))
}

// GetKDCresourcetypeconstraineddelegationTGSRequests gets the value of KDCresourcetypeconstraineddelegationTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCresourcetypeconstraineddelegationTGSRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCresourcetypeconstraineddelegationTGSRequests")
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

// SetKDCTGSRequests sets the value of KDCTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKDCTGSRequests(value uint32) (err error) {
	return instance.SetProperty("KDCTGSRequests", (value))
}

// GetKDCTGSRequests gets the value of KDCTGSRequests for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKDCTGSRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KDCTGSRequests")
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

// SetKerberosAuthentications sets the value of KerberosAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyKerberosAuthentications(value uint32) (err error) {
	return instance.SetProperty("KerberosAuthentications", (value))
}

// GetKerberosAuthentications gets the value of KerberosAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyKerberosAuthentications() (value uint32, err error) {
	retValue, err := instance.GetProperty("KerberosAuthentications")
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

// SetNTLMAuthentications sets the value of NTLMAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertyNTLMAuthentications(value uint32) (err error) {
	return instance.SetProperty("NTLMAuthentications", (value))
}

// GetNTLMAuthentications gets the value of NTLMAuthentications for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertyNTLMAuthentications() (value uint32, err error) {
	retValue, err := instance.GetProperty("NTLMAuthentications")
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

// SetSchannelSessionCacheEntries sets the value of SchannelSessionCacheEntries for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertySchannelSessionCacheEntries(value uint32) (err error) {
	return instance.SetProperty("SchannelSessionCacheEntries", (value))
}

// GetSchannelSessionCacheEntries gets the value of SchannelSessionCacheEntries for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertySchannelSessionCacheEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("SchannelSessionCacheEntries")
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

// SetSSLClientSideFullHandshakes sets the value of SSLClientSideFullHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertySSLClientSideFullHandshakes(value uint32) (err error) {
	return instance.SetProperty("SSLClientSideFullHandshakes", (value))
}

// GetSSLClientSideFullHandshakes gets the value of SSLClientSideFullHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertySSLClientSideFullHandshakes() (value uint32, err error) {
	retValue, err := instance.GetProperty("SSLClientSideFullHandshakes")
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

// SetSSLClientSideReconnectHandshakes sets the value of SSLClientSideReconnectHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertySSLClientSideReconnectHandshakes(value uint32) (err error) {
	return instance.SetProperty("SSLClientSideReconnectHandshakes", (value))
}

// GetSSLClientSideReconnectHandshakes gets the value of SSLClientSideReconnectHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertySSLClientSideReconnectHandshakes() (value uint32, err error) {
	retValue, err := instance.GetProperty("SSLClientSideReconnectHandshakes")
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

// SetSSLServerSideFullHandshakes sets the value of SSLServerSideFullHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertySSLServerSideFullHandshakes(value uint32) (err error) {
	return instance.SetProperty("SSLServerSideFullHandshakes", (value))
}

// GetSSLServerSideFullHandshakes gets the value of SSLServerSideFullHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertySSLServerSideFullHandshakes() (value uint32, err error) {
	retValue, err := instance.GetProperty("SSLServerSideFullHandshakes")
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

// SetSSLServerSideReconnectHandshakes sets the value of SSLServerSideReconnectHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) SetPropertySSLServerSideReconnectHandshakes(value uint32) (err error) {
	return instance.SetProperty("SSLServerSideReconnectHandshakes", (value))
}

// GetSSLServerSideReconnectHandshakes gets the value of SSLServerSideReconnectHandshakes for the instance
func (instance *Win32_PerfRawData_Lsa_SecuritySystemWideStatistics) GetPropertySSLServerSideReconnectHandshakes() (value uint32, err error) {
	retValue, err := instance.GetProperty("SSLServerSideReconnectHandshakes")
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
