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

// SoftwareLicensingService struct
type SoftwareLicensingService struct {
	*cim.WmiInstance

	//
	ClientMachineID string

	//
	DiscoveredKeyManagementServiceMachineIpAddress string

	//
	DiscoveredKeyManagementServiceMachineName string

	//
	DiscoveredKeyManagementServiceMachinePort uint32

	//
	IsKeyManagementServiceMachine uint32

	//
	KeyManagementServiceCurrentCount uint32

	//
	KeyManagementServiceDnsPublishing bool

	//
	KeyManagementServiceFailedRequests uint32

	//
	KeyManagementServiceHostCaching bool

	//
	KeyManagementServiceLicensedRequests uint32

	//
	KeyManagementServiceListeningPort uint32

	//
	KeyManagementServiceLookupDomain string

	//
	KeyManagementServiceLowPriority bool

	//
	KeyManagementServiceMachine string

	//
	KeyManagementServiceNonGenuineGraceRequests uint32

	//
	KeyManagementServiceNotificationRequests uint32

	//
	KeyManagementServiceOOBGraceRequests uint32

	//
	KeyManagementServiceOOTGraceRequests uint32

	//
	KeyManagementServicePort uint32

	//
	KeyManagementServiceProductKeyID string

	//
	KeyManagementServiceTotalRequests uint32

	//
	KeyManagementServiceUnlicensedRequests uint32

	//
	OA2xBiosMarkerMinorVersion uint32

	//
	OA2xBiosMarkerStatus uint32

	//
	OA3xOriginalProductKey string

	//
	OA3xOriginalProductKeyDescription string

	//
	OA3xOriginalProductKeyPkPn string

	//
	PolicyCacheRefreshRequired uint32

	//
	RemainingWindowsReArmCount uint32

	//
	RequiredClientCount uint32

	//
	TokenActivationAdditionalInfo string

	//
	TokenActivationCertificateThumbprint string

	//
	TokenActivationGrantNumber uint32

	//
	TokenActivationILID string

	//
	TokenActivationILVID uint32

	//
	Version string

	//
	VLActivationInterval uint32

	//
	VLRenewalInterval uint32
}

func NewSoftwareLicensingServiceEx1(instance *cim.WmiInstance) (newInstance *SoftwareLicensingService, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &SoftwareLicensingService{
		WmiInstance: tmp,
	}
	return
}

func NewSoftwareLicensingServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *SoftwareLicensingService, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &SoftwareLicensingService{
		WmiInstance: tmp,
	}
	return
}

// SetClientMachineID sets the value of ClientMachineID for the instance
func (instance *SoftwareLicensingService) SetPropertyClientMachineID(value string) (err error) {
	return instance.SetProperty("ClientMachineID", (value))
}

// GetClientMachineID gets the value of ClientMachineID for the instance
func (instance *SoftwareLicensingService) GetPropertyClientMachineID() (value string, err error) {
	retValue, err := instance.GetProperty("ClientMachineID")
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

// SetDiscoveredKeyManagementServiceMachineIpAddress sets the value of DiscoveredKeyManagementServiceMachineIpAddress for the instance
func (instance *SoftwareLicensingService) SetPropertyDiscoveredKeyManagementServiceMachineIpAddress(value string) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachineIpAddress", (value))
}

// GetDiscoveredKeyManagementServiceMachineIpAddress gets the value of DiscoveredKeyManagementServiceMachineIpAddress for the instance
func (instance *SoftwareLicensingService) GetPropertyDiscoveredKeyManagementServiceMachineIpAddress() (value string, err error) {
	retValue, err := instance.GetProperty("DiscoveredKeyManagementServiceMachineIpAddress")
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

// SetDiscoveredKeyManagementServiceMachineName sets the value of DiscoveredKeyManagementServiceMachineName for the instance
func (instance *SoftwareLicensingService) SetPropertyDiscoveredKeyManagementServiceMachineName(value string) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachineName", (value))
}

// GetDiscoveredKeyManagementServiceMachineName gets the value of DiscoveredKeyManagementServiceMachineName for the instance
func (instance *SoftwareLicensingService) GetPropertyDiscoveredKeyManagementServiceMachineName() (value string, err error) {
	retValue, err := instance.GetProperty("DiscoveredKeyManagementServiceMachineName")
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

// SetDiscoveredKeyManagementServiceMachinePort sets the value of DiscoveredKeyManagementServiceMachinePort for the instance
func (instance *SoftwareLicensingService) SetPropertyDiscoveredKeyManagementServiceMachinePort(value uint32) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachinePort", (value))
}

// GetDiscoveredKeyManagementServiceMachinePort gets the value of DiscoveredKeyManagementServiceMachinePort for the instance
func (instance *SoftwareLicensingService) GetPropertyDiscoveredKeyManagementServiceMachinePort() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiscoveredKeyManagementServiceMachinePort")
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

// SetIsKeyManagementServiceMachine sets the value of IsKeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingService) SetPropertyIsKeyManagementServiceMachine(value uint32) (err error) {
	return instance.SetProperty("IsKeyManagementServiceMachine", (value))
}

// GetIsKeyManagementServiceMachine gets the value of IsKeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingService) GetPropertyIsKeyManagementServiceMachine() (value uint32, err error) {
	retValue, err := instance.GetProperty("IsKeyManagementServiceMachine")
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

// SetKeyManagementServiceCurrentCount sets the value of KeyManagementServiceCurrentCount for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceCurrentCount(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceCurrentCount", (value))
}

// GetKeyManagementServiceCurrentCount gets the value of KeyManagementServiceCurrentCount for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceCurrentCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceCurrentCount")
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

// SetKeyManagementServiceDnsPublishing sets the value of KeyManagementServiceDnsPublishing for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceDnsPublishing(value bool) (err error) {
	return instance.SetProperty("KeyManagementServiceDnsPublishing", (value))
}

// GetKeyManagementServiceDnsPublishing gets the value of KeyManagementServiceDnsPublishing for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceDnsPublishing() (value bool, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceDnsPublishing")
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

// SetKeyManagementServiceFailedRequests sets the value of KeyManagementServiceFailedRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceFailedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceFailedRequests", (value))
}

// GetKeyManagementServiceFailedRequests gets the value of KeyManagementServiceFailedRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceFailedRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceFailedRequests")
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

// SetKeyManagementServiceHostCaching sets the value of KeyManagementServiceHostCaching for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceHostCaching(value bool) (err error) {
	return instance.SetProperty("KeyManagementServiceHostCaching", (value))
}

// GetKeyManagementServiceHostCaching gets the value of KeyManagementServiceHostCaching for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceHostCaching() (value bool, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceHostCaching")
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

// SetKeyManagementServiceLicensedRequests sets the value of KeyManagementServiceLicensedRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceLicensedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceLicensedRequests", (value))
}

// GetKeyManagementServiceLicensedRequests gets the value of KeyManagementServiceLicensedRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceLicensedRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceLicensedRequests")
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

// SetKeyManagementServiceListeningPort sets the value of KeyManagementServiceListeningPort for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceListeningPort(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceListeningPort", (value))
}

// GetKeyManagementServiceListeningPort gets the value of KeyManagementServiceListeningPort for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceListeningPort() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceListeningPort")
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

// SetKeyManagementServiceLookupDomain sets the value of KeyManagementServiceLookupDomain for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceLookupDomain(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceLookupDomain", (value))
}

// GetKeyManagementServiceLookupDomain gets the value of KeyManagementServiceLookupDomain for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceLookupDomain() (value string, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceLookupDomain")
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

// SetKeyManagementServiceLowPriority sets the value of KeyManagementServiceLowPriority for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceLowPriority(value bool) (err error) {
	return instance.SetProperty("KeyManagementServiceLowPriority", (value))
}

// GetKeyManagementServiceLowPriority gets the value of KeyManagementServiceLowPriority for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceLowPriority() (value bool, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceLowPriority")
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

// SetKeyManagementServiceMachine sets the value of KeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceMachine(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceMachine", (value))
}

// GetKeyManagementServiceMachine gets the value of KeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceMachine() (value string, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceMachine")
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

// SetKeyManagementServiceNonGenuineGraceRequests sets the value of KeyManagementServiceNonGenuineGraceRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceNonGenuineGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceNonGenuineGraceRequests", (value))
}

// GetKeyManagementServiceNonGenuineGraceRequests gets the value of KeyManagementServiceNonGenuineGraceRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceNonGenuineGraceRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceNonGenuineGraceRequests")
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

// SetKeyManagementServiceNotificationRequests sets the value of KeyManagementServiceNotificationRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceNotificationRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceNotificationRequests", (value))
}

// GetKeyManagementServiceNotificationRequests gets the value of KeyManagementServiceNotificationRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceNotificationRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceNotificationRequests")
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

// SetKeyManagementServiceOOBGraceRequests sets the value of KeyManagementServiceOOBGraceRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceOOBGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceOOBGraceRequests", (value))
}

// GetKeyManagementServiceOOBGraceRequests gets the value of KeyManagementServiceOOBGraceRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceOOBGraceRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceOOBGraceRequests")
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

// SetKeyManagementServiceOOTGraceRequests sets the value of KeyManagementServiceOOTGraceRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceOOTGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceOOTGraceRequests", (value))
}

// GetKeyManagementServiceOOTGraceRequests gets the value of KeyManagementServiceOOTGraceRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceOOTGraceRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceOOTGraceRequests")
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

// SetKeyManagementServicePort sets the value of KeyManagementServicePort for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServicePort(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServicePort", (value))
}

// GetKeyManagementServicePort gets the value of KeyManagementServicePort for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServicePort() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServicePort")
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

// SetKeyManagementServiceProductKeyID sets the value of KeyManagementServiceProductKeyID for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceProductKeyID(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceProductKeyID", (value))
}

// GetKeyManagementServiceProductKeyID gets the value of KeyManagementServiceProductKeyID for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceProductKeyID() (value string, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceProductKeyID")
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

// SetKeyManagementServiceTotalRequests sets the value of KeyManagementServiceTotalRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceTotalRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceTotalRequests", (value))
}

// GetKeyManagementServiceTotalRequests gets the value of KeyManagementServiceTotalRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceTotalRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceTotalRequests")
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

// SetKeyManagementServiceUnlicensedRequests sets the value of KeyManagementServiceUnlicensedRequests for the instance
func (instance *SoftwareLicensingService) SetPropertyKeyManagementServiceUnlicensedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceUnlicensedRequests", (value))
}

// GetKeyManagementServiceUnlicensedRequests gets the value of KeyManagementServiceUnlicensedRequests for the instance
func (instance *SoftwareLicensingService) GetPropertyKeyManagementServiceUnlicensedRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeyManagementServiceUnlicensedRequests")
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

// SetOA2xBiosMarkerMinorVersion sets the value of OA2xBiosMarkerMinorVersion for the instance
func (instance *SoftwareLicensingService) SetPropertyOA2xBiosMarkerMinorVersion(value uint32) (err error) {
	return instance.SetProperty("OA2xBiosMarkerMinorVersion", (value))
}

// GetOA2xBiosMarkerMinorVersion gets the value of OA2xBiosMarkerMinorVersion for the instance
func (instance *SoftwareLicensingService) GetPropertyOA2xBiosMarkerMinorVersion() (value uint32, err error) {
	retValue, err := instance.GetProperty("OA2xBiosMarkerMinorVersion")
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

// SetOA2xBiosMarkerStatus sets the value of OA2xBiosMarkerStatus for the instance
func (instance *SoftwareLicensingService) SetPropertyOA2xBiosMarkerStatus(value uint32) (err error) {
	return instance.SetProperty("OA2xBiosMarkerStatus", (value))
}

// GetOA2xBiosMarkerStatus gets the value of OA2xBiosMarkerStatus for the instance
func (instance *SoftwareLicensingService) GetPropertyOA2xBiosMarkerStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("OA2xBiosMarkerStatus")
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

// SetOA3xOriginalProductKey sets the value of OA3xOriginalProductKey for the instance
func (instance *SoftwareLicensingService) SetPropertyOA3xOriginalProductKey(value string) (err error) {
	return instance.SetProperty("OA3xOriginalProductKey", (value))
}

// GetOA3xOriginalProductKey gets the value of OA3xOriginalProductKey for the instance
func (instance *SoftwareLicensingService) GetPropertyOA3xOriginalProductKey() (value string, err error) {
	retValue, err := instance.GetProperty("OA3xOriginalProductKey")
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

// SetOA3xOriginalProductKeyDescription sets the value of OA3xOriginalProductKeyDescription for the instance
func (instance *SoftwareLicensingService) SetPropertyOA3xOriginalProductKeyDescription(value string) (err error) {
	return instance.SetProperty("OA3xOriginalProductKeyDescription", (value))
}

// GetOA3xOriginalProductKeyDescription gets the value of OA3xOriginalProductKeyDescription for the instance
func (instance *SoftwareLicensingService) GetPropertyOA3xOriginalProductKeyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OA3xOriginalProductKeyDescription")
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

// SetOA3xOriginalProductKeyPkPn sets the value of OA3xOriginalProductKeyPkPn for the instance
func (instance *SoftwareLicensingService) SetPropertyOA3xOriginalProductKeyPkPn(value string) (err error) {
	return instance.SetProperty("OA3xOriginalProductKeyPkPn", (value))
}

// GetOA3xOriginalProductKeyPkPn gets the value of OA3xOriginalProductKeyPkPn for the instance
func (instance *SoftwareLicensingService) GetPropertyOA3xOriginalProductKeyPkPn() (value string, err error) {
	retValue, err := instance.GetProperty("OA3xOriginalProductKeyPkPn")
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

// SetPolicyCacheRefreshRequired sets the value of PolicyCacheRefreshRequired for the instance
func (instance *SoftwareLicensingService) SetPropertyPolicyCacheRefreshRequired(value uint32) (err error) {
	return instance.SetProperty("PolicyCacheRefreshRequired", (value))
}

// GetPolicyCacheRefreshRequired gets the value of PolicyCacheRefreshRequired for the instance
func (instance *SoftwareLicensingService) GetPropertyPolicyCacheRefreshRequired() (value uint32, err error) {
	retValue, err := instance.GetProperty("PolicyCacheRefreshRequired")
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

// SetRemainingWindowsReArmCount sets the value of RemainingWindowsReArmCount for the instance
func (instance *SoftwareLicensingService) SetPropertyRemainingWindowsReArmCount(value uint32) (err error) {
	return instance.SetProperty("RemainingWindowsReArmCount", (value))
}

// GetRemainingWindowsReArmCount gets the value of RemainingWindowsReArmCount for the instance
func (instance *SoftwareLicensingService) GetPropertyRemainingWindowsReArmCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemainingWindowsReArmCount")
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

// SetRequiredClientCount sets the value of RequiredClientCount for the instance
func (instance *SoftwareLicensingService) SetPropertyRequiredClientCount(value uint32) (err error) {
	return instance.SetProperty("RequiredClientCount", (value))
}

// GetRequiredClientCount gets the value of RequiredClientCount for the instance
func (instance *SoftwareLicensingService) GetPropertyRequiredClientCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequiredClientCount")
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

// SetTokenActivationAdditionalInfo sets the value of TokenActivationAdditionalInfo for the instance
func (instance *SoftwareLicensingService) SetPropertyTokenActivationAdditionalInfo(value string) (err error) {
	return instance.SetProperty("TokenActivationAdditionalInfo", (value))
}

// GetTokenActivationAdditionalInfo gets the value of TokenActivationAdditionalInfo for the instance
func (instance *SoftwareLicensingService) GetPropertyTokenActivationAdditionalInfo() (value string, err error) {
	retValue, err := instance.GetProperty("TokenActivationAdditionalInfo")
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

// SetTokenActivationCertificateThumbprint sets the value of TokenActivationCertificateThumbprint for the instance
func (instance *SoftwareLicensingService) SetPropertyTokenActivationCertificateThumbprint(value string) (err error) {
	return instance.SetProperty("TokenActivationCertificateThumbprint", (value))
}

// GetTokenActivationCertificateThumbprint gets the value of TokenActivationCertificateThumbprint for the instance
func (instance *SoftwareLicensingService) GetPropertyTokenActivationCertificateThumbprint() (value string, err error) {
	retValue, err := instance.GetProperty("TokenActivationCertificateThumbprint")
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

// SetTokenActivationGrantNumber sets the value of TokenActivationGrantNumber for the instance
func (instance *SoftwareLicensingService) SetPropertyTokenActivationGrantNumber(value uint32) (err error) {
	return instance.SetProperty("TokenActivationGrantNumber", (value))
}

// GetTokenActivationGrantNumber gets the value of TokenActivationGrantNumber for the instance
func (instance *SoftwareLicensingService) GetPropertyTokenActivationGrantNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("TokenActivationGrantNumber")
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

// SetTokenActivationILID sets the value of TokenActivationILID for the instance
func (instance *SoftwareLicensingService) SetPropertyTokenActivationILID(value string) (err error) {
	return instance.SetProperty("TokenActivationILID", (value))
}

// GetTokenActivationILID gets the value of TokenActivationILID for the instance
func (instance *SoftwareLicensingService) GetPropertyTokenActivationILID() (value string, err error) {
	retValue, err := instance.GetProperty("TokenActivationILID")
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

// SetTokenActivationILVID sets the value of TokenActivationILVID for the instance
func (instance *SoftwareLicensingService) SetPropertyTokenActivationILVID(value uint32) (err error) {
	return instance.SetProperty("TokenActivationILVID", (value))
}

// GetTokenActivationILVID gets the value of TokenActivationILVID for the instance
func (instance *SoftwareLicensingService) GetPropertyTokenActivationILVID() (value uint32, err error) {
	retValue, err := instance.GetProperty("TokenActivationILVID")
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

// SetVersion sets the value of Version for the instance
func (instance *SoftwareLicensingService) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *SoftwareLicensingService) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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

// SetVLActivationInterval sets the value of VLActivationInterval for the instance
func (instance *SoftwareLicensingService) SetPropertyVLActivationInterval(value uint32) (err error) {
	return instance.SetProperty("VLActivationInterval", (value))
}

// GetVLActivationInterval gets the value of VLActivationInterval for the instance
func (instance *SoftwareLicensingService) GetPropertyVLActivationInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("VLActivationInterval")
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

// SetVLRenewalInterval sets the value of VLRenewalInterval for the instance
func (instance *SoftwareLicensingService) SetPropertyVLRenewalInterval(value uint32) (err error) {
	return instance.SetProperty("VLRenewalInterval", (value))
}

// GetVLRenewalInterval gets the value of VLRenewalInterval for the instance
func (instance *SoftwareLicensingService) GetPropertyVLRenewalInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("VLRenewalInterval")
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

//

// <param name="ProductKey" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) InstallProductKey( /* IN */ ProductKey string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("InstallProductKey", ProductKey)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="License" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) InstallLicense( /* IN */ License string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("InstallLicense", License)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="LicensePackage" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) InstallLicensePackage( /* IN */ LicensePackage string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("InstallLicensePackage", LicensePackage)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="MachineName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetKeyManagementServiceMachine( /* IN */ MachineName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServiceMachine", MachineName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearKeyManagementServiceMachine() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearKeyManagementServiceMachine")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PortNumber" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetKeyManagementServicePort( /* IN */ PortNumber uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServicePort", PortNumber)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearKeyManagementServicePort() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearKeyManagementServicePort")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="LookupDomain" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetKeyManagementServiceLookupDomain( /* IN */ LookupDomain string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServiceLookupDomain", LookupDomain)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearKeyManagementServiceLookupDomain() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearKeyManagementServiceLookupDomain")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ActivationInterval" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetVLActivationInterval( /* IN */ ActivationInterval uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVLActivationInterval", ActivationInterval)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="RenewalInterval" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetVLRenewalInterval( /* IN */ RenewalInterval uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVLRenewalInterval", RenewalInterval)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearProductKeyFromRegistry() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearProductKeyFromRegistry")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ServerUrl" type="string "></param>
// <param name="TemplateId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) AcquireGenuineTicket( /* IN */ TemplateId string,
	/* IN */ ServerUrl string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AcquireGenuineTicket", TemplateId, ServerUrl)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ReArmWindows() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ReArmWindows")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ApplicationId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ReArmApp( /* IN */ ApplicationId string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ReArmApp", ApplicationId)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) RefreshLicenseStatus() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("RefreshLicenseStatus")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PortNumber" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetKeyManagementServiceListeningPort( /* IN */ PortNumber uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServiceListeningPort", PortNumber)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearKeyManagementServiceListeningPort() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearKeyManagementServiceListeningPort")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DisablePublishing" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) DisableKeyManagementServiceDnsPublishing( /* IN */ DisablePublishing bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DisableKeyManagementServiceDnsPublishing", DisablePublishing)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="EnableLowPriority" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) EnableKeyManagementServiceLowPriority( /* IN */ EnableLowPriority bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableKeyManagementServiceLowPriority", EnableLowPriority)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DisableCaching" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) DisableKeyManagementServiceHostCaching( /* IN */ DisableCaching bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DisableKeyManagementServiceHostCaching", DisableCaching)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ProductKey" type="string "></param>

// <param name="InstallationID" type="string "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) GenerateActiveDirectoryOfflineActivationId( /* IN */ ProductKey string,
	/* OUT */ InstallationID string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GenerateActiveDirectoryOfflineActivationId", ProductKey)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ActivationObjectName" type="string "></param>
// <param name="ConfirmationID" type="string "></param>
// <param name="ProductKey" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) DepositActiveDirectoryOfflineActivationConfirmation( /* IN */ ProductKey string,
	/* IN */ ConfirmationID string,
	/* IN */ ActivationObjectName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DepositActiveDirectoryOfflineActivationConfirmation", ProductKey, ConfirmationID, ActivationObjectName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ActivationObjectName" type="string "></param>
// <param name="ProductKey" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) DoActiveDirectoryOnlineActivation( /* IN */ ProductKey string,
	/* IN */ ActivationObjectName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DoActiveDirectoryOnlineActivation", ProductKey, ActivationObjectName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ActivationType" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) SetVLActivationTypeEnabled( /* IN */ ActivationType uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVLActivationTypeEnabled", ActivationType)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingService) ClearVLActivationTypeEnabled() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearVLActivationTypeEnabled")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
