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

// SoftwareLicensingProduct struct
type SoftwareLicensingProduct struct {
	*cim.WmiInstance

	//
	ADActivationCsvlkPid string

	//
	ADActivationCsvlkSkuId string

	//
	ADActivationObjectDN string

	//
	ADActivationObjectName string

	//
	ApplicationID string

	//
	AutomaticVMActivationHostDigitalPid2 string

	//
	AutomaticVMActivationHostMachineName string

	//
	AutomaticVMActivationLastActivationTime string

	//
	Description string

	//
	DiscoveredKeyManagementServiceMachineIpAddress string

	//
	DiscoveredKeyManagementServiceMachineName string

	//
	DiscoveredKeyManagementServiceMachinePort uint32

	//
	EvaluationEndDate string

	//
	ExtendedGrace uint32

	//
	GenuineStatus uint32

	//
	GracePeriodRemaining uint32

	//
	IAID string

	//
	ID string

	//
	IsKeyManagementServiceMachine uint32

	//
	KeyManagementServiceCurrentCount uint32

	//
	KeyManagementServiceFailedRequests uint32

	//
	KeyManagementServiceLicensedRequests uint32

	//
	KeyManagementServiceLookupDomain string

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
	LicenseDependsOn string

	//
	LicenseFamily string

	//
	LicenseIsAddon bool

	//
	LicenseStatus uint32

	//
	LicenseStatusReason uint32

	//
	MachineURL string

	//
	Name string

	//
	OfflineInstallationId string

	//
	PartialProductKey string

	//
	ProcessorURL string

	//
	ProductKeyChannel string

	//
	ProductKeyID string

	//
	ProductKeyID2 string

	//
	ProductKeyURL string

	//
	RemainingAppReArmCount uint32

	//
	RemainingSkuReArmCount uint32

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
	TrustedTime string

	//
	UseLicenseURL string

	//
	ValidationURL string

	//
	VLActivationInterval uint32

	//
	VLActivationType uint32

	//
	VLActivationTypeEnabled uint32

	//
	VLRenewalInterval uint32
}

func NewSoftwareLicensingProductEx1(instance *cim.WmiInstance) (newInstance *SoftwareLicensingProduct, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &SoftwareLicensingProduct{
		WmiInstance: tmp,
	}
	return
}

func NewSoftwareLicensingProductEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *SoftwareLicensingProduct, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &SoftwareLicensingProduct{
		WmiInstance: tmp,
	}
	return
}

// SetADActivationCsvlkPid sets the value of ADActivationCsvlkPid for the instance
func (instance *SoftwareLicensingProduct) SetPropertyADActivationCsvlkPid(value string) (err error) {
	return instance.SetProperty("ADActivationCsvlkPid", (value))
}

// GetADActivationCsvlkPid gets the value of ADActivationCsvlkPid for the instance
func (instance *SoftwareLicensingProduct) GetPropertyADActivationCsvlkPid() (value string, err error) {
	retValue, err := instance.GetProperty("ADActivationCsvlkPid")
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

// SetADActivationCsvlkSkuId sets the value of ADActivationCsvlkSkuId for the instance
func (instance *SoftwareLicensingProduct) SetPropertyADActivationCsvlkSkuId(value string) (err error) {
	return instance.SetProperty("ADActivationCsvlkSkuId", (value))
}

// GetADActivationCsvlkSkuId gets the value of ADActivationCsvlkSkuId for the instance
func (instance *SoftwareLicensingProduct) GetPropertyADActivationCsvlkSkuId() (value string, err error) {
	retValue, err := instance.GetProperty("ADActivationCsvlkSkuId")
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

// SetADActivationObjectDN sets the value of ADActivationObjectDN for the instance
func (instance *SoftwareLicensingProduct) SetPropertyADActivationObjectDN(value string) (err error) {
	return instance.SetProperty("ADActivationObjectDN", (value))
}

// GetADActivationObjectDN gets the value of ADActivationObjectDN for the instance
func (instance *SoftwareLicensingProduct) GetPropertyADActivationObjectDN() (value string, err error) {
	retValue, err := instance.GetProperty("ADActivationObjectDN")
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

// SetADActivationObjectName sets the value of ADActivationObjectName for the instance
func (instance *SoftwareLicensingProduct) SetPropertyADActivationObjectName(value string) (err error) {
	return instance.SetProperty("ADActivationObjectName", (value))
}

// GetADActivationObjectName gets the value of ADActivationObjectName for the instance
func (instance *SoftwareLicensingProduct) GetPropertyADActivationObjectName() (value string, err error) {
	retValue, err := instance.GetProperty("ADActivationObjectName")
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

// SetApplicationID sets the value of ApplicationID for the instance
func (instance *SoftwareLicensingProduct) SetPropertyApplicationID(value string) (err error) {
	return instance.SetProperty("ApplicationID", (value))
}

// GetApplicationID gets the value of ApplicationID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyApplicationID() (value string, err error) {
	retValue, err := instance.GetProperty("ApplicationID")
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

// SetAutomaticVMActivationHostDigitalPid2 sets the value of AutomaticVMActivationHostDigitalPid2 for the instance
func (instance *SoftwareLicensingProduct) SetPropertyAutomaticVMActivationHostDigitalPid2(value string) (err error) {
	return instance.SetProperty("AutomaticVMActivationHostDigitalPid2", (value))
}

// GetAutomaticVMActivationHostDigitalPid2 gets the value of AutomaticVMActivationHostDigitalPid2 for the instance
func (instance *SoftwareLicensingProduct) GetPropertyAutomaticVMActivationHostDigitalPid2() (value string, err error) {
	retValue, err := instance.GetProperty("AutomaticVMActivationHostDigitalPid2")
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

// SetAutomaticVMActivationHostMachineName sets the value of AutomaticVMActivationHostMachineName for the instance
func (instance *SoftwareLicensingProduct) SetPropertyAutomaticVMActivationHostMachineName(value string) (err error) {
	return instance.SetProperty("AutomaticVMActivationHostMachineName", (value))
}

// GetAutomaticVMActivationHostMachineName gets the value of AutomaticVMActivationHostMachineName for the instance
func (instance *SoftwareLicensingProduct) GetPropertyAutomaticVMActivationHostMachineName() (value string, err error) {
	retValue, err := instance.GetProperty("AutomaticVMActivationHostMachineName")
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

// SetAutomaticVMActivationLastActivationTime sets the value of AutomaticVMActivationLastActivationTime for the instance
func (instance *SoftwareLicensingProduct) SetPropertyAutomaticVMActivationLastActivationTime(value string) (err error) {
	return instance.SetProperty("AutomaticVMActivationLastActivationTime", (value))
}

// GetAutomaticVMActivationLastActivationTime gets the value of AutomaticVMActivationLastActivationTime for the instance
func (instance *SoftwareLicensingProduct) GetPropertyAutomaticVMActivationLastActivationTime() (value string, err error) {
	retValue, err := instance.GetProperty("AutomaticVMActivationLastActivationTime")
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

// SetDescription sets the value of Description for the instance
func (instance *SoftwareLicensingProduct) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *SoftwareLicensingProduct) GetPropertyDescription() (value string, err error) {
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

// SetDiscoveredKeyManagementServiceMachineIpAddress sets the value of DiscoveredKeyManagementServiceMachineIpAddress for the instance
func (instance *SoftwareLicensingProduct) SetPropertyDiscoveredKeyManagementServiceMachineIpAddress(value string) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachineIpAddress", (value))
}

// GetDiscoveredKeyManagementServiceMachineIpAddress gets the value of DiscoveredKeyManagementServiceMachineIpAddress for the instance
func (instance *SoftwareLicensingProduct) GetPropertyDiscoveredKeyManagementServiceMachineIpAddress() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyDiscoveredKeyManagementServiceMachineName(value string) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachineName", (value))
}

// GetDiscoveredKeyManagementServiceMachineName gets the value of DiscoveredKeyManagementServiceMachineName for the instance
func (instance *SoftwareLicensingProduct) GetPropertyDiscoveredKeyManagementServiceMachineName() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyDiscoveredKeyManagementServiceMachinePort(value uint32) (err error) {
	return instance.SetProperty("DiscoveredKeyManagementServiceMachinePort", (value))
}

// GetDiscoveredKeyManagementServiceMachinePort gets the value of DiscoveredKeyManagementServiceMachinePort for the instance
func (instance *SoftwareLicensingProduct) GetPropertyDiscoveredKeyManagementServiceMachinePort() (value uint32, err error) {
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

// SetEvaluationEndDate sets the value of EvaluationEndDate for the instance
func (instance *SoftwareLicensingProduct) SetPropertyEvaluationEndDate(value string) (err error) {
	return instance.SetProperty("EvaluationEndDate", (value))
}

// GetEvaluationEndDate gets the value of EvaluationEndDate for the instance
func (instance *SoftwareLicensingProduct) GetPropertyEvaluationEndDate() (value string, err error) {
	retValue, err := instance.GetProperty("EvaluationEndDate")
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

// SetExtendedGrace sets the value of ExtendedGrace for the instance
func (instance *SoftwareLicensingProduct) SetPropertyExtendedGrace(value uint32) (err error) {
	return instance.SetProperty("ExtendedGrace", (value))
}

// GetExtendedGrace gets the value of ExtendedGrace for the instance
func (instance *SoftwareLicensingProduct) GetPropertyExtendedGrace() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExtendedGrace")
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

// SetGenuineStatus sets the value of GenuineStatus for the instance
func (instance *SoftwareLicensingProduct) SetPropertyGenuineStatus(value uint32) (err error) {
	return instance.SetProperty("GenuineStatus", (value))
}

// GetGenuineStatus gets the value of GenuineStatus for the instance
func (instance *SoftwareLicensingProduct) GetPropertyGenuineStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("GenuineStatus")
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

// SetGracePeriodRemaining sets the value of GracePeriodRemaining for the instance
func (instance *SoftwareLicensingProduct) SetPropertyGracePeriodRemaining(value uint32) (err error) {
	return instance.SetProperty("GracePeriodRemaining", (value))
}

// GetGracePeriodRemaining gets the value of GracePeriodRemaining for the instance
func (instance *SoftwareLicensingProduct) GetPropertyGracePeriodRemaining() (value uint32, err error) {
	retValue, err := instance.GetProperty("GracePeriodRemaining")
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

// SetIAID sets the value of IAID for the instance
func (instance *SoftwareLicensingProduct) SetPropertyIAID(value string) (err error) {
	return instance.SetProperty("IAID", (value))
}

// GetIAID gets the value of IAID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyIAID() (value string, err error) {
	retValue, err := instance.GetProperty("IAID")
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

// SetID sets the value of ID for the instance
func (instance *SoftwareLicensingProduct) SetPropertyID(value string) (err error) {
	return instance.SetProperty("ID", (value))
}

// GetID gets the value of ID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyID() (value string, err error) {
	retValue, err := instance.GetProperty("ID")
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

// SetIsKeyManagementServiceMachine sets the value of IsKeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingProduct) SetPropertyIsKeyManagementServiceMachine(value uint32) (err error) {
	return instance.SetProperty("IsKeyManagementServiceMachine", (value))
}

// GetIsKeyManagementServiceMachine gets the value of IsKeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingProduct) GetPropertyIsKeyManagementServiceMachine() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceCurrentCount(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceCurrentCount", (value))
}

// GetKeyManagementServiceCurrentCount gets the value of KeyManagementServiceCurrentCount for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceCurrentCount() (value uint32, err error) {
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

// SetKeyManagementServiceFailedRequests sets the value of KeyManagementServiceFailedRequests for the instance
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceFailedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceFailedRequests", (value))
}

// GetKeyManagementServiceFailedRequests gets the value of KeyManagementServiceFailedRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceFailedRequests() (value uint32, err error) {
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

// SetKeyManagementServiceLicensedRequests sets the value of KeyManagementServiceLicensedRequests for the instance
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceLicensedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceLicensedRequests", (value))
}

// GetKeyManagementServiceLicensedRequests gets the value of KeyManagementServiceLicensedRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceLicensedRequests() (value uint32, err error) {
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

// SetKeyManagementServiceLookupDomain sets the value of KeyManagementServiceLookupDomain for the instance
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceLookupDomain(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceLookupDomain", (value))
}

// GetKeyManagementServiceLookupDomain gets the value of KeyManagementServiceLookupDomain for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceLookupDomain() (value string, err error) {
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

// SetKeyManagementServiceMachine sets the value of KeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceMachine(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceMachine", (value))
}

// GetKeyManagementServiceMachine gets the value of KeyManagementServiceMachine for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceMachine() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceNonGenuineGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceNonGenuineGraceRequests", (value))
}

// GetKeyManagementServiceNonGenuineGraceRequests gets the value of KeyManagementServiceNonGenuineGraceRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceNonGenuineGraceRequests() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceNotificationRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceNotificationRequests", (value))
}

// GetKeyManagementServiceNotificationRequests gets the value of KeyManagementServiceNotificationRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceNotificationRequests() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceOOBGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceOOBGraceRequests", (value))
}

// GetKeyManagementServiceOOBGraceRequests gets the value of KeyManagementServiceOOBGraceRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceOOBGraceRequests() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceOOTGraceRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceOOTGraceRequests", (value))
}

// GetKeyManagementServiceOOTGraceRequests gets the value of KeyManagementServiceOOTGraceRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceOOTGraceRequests() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServicePort(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServicePort", (value))
}

// GetKeyManagementServicePort gets the value of KeyManagementServicePort for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServicePort() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceProductKeyID(value string) (err error) {
	return instance.SetProperty("KeyManagementServiceProductKeyID", (value))
}

// GetKeyManagementServiceProductKeyID gets the value of KeyManagementServiceProductKeyID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceProductKeyID() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceTotalRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceTotalRequests", (value))
}

// GetKeyManagementServiceTotalRequests gets the value of KeyManagementServiceTotalRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceTotalRequests() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyKeyManagementServiceUnlicensedRequests(value uint32) (err error) {
	return instance.SetProperty("KeyManagementServiceUnlicensedRequests", (value))
}

// GetKeyManagementServiceUnlicensedRequests gets the value of KeyManagementServiceUnlicensedRequests for the instance
func (instance *SoftwareLicensingProduct) GetPropertyKeyManagementServiceUnlicensedRequests() (value uint32, err error) {
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

// SetLicenseDependsOn sets the value of LicenseDependsOn for the instance
func (instance *SoftwareLicensingProduct) SetPropertyLicenseDependsOn(value string) (err error) {
	return instance.SetProperty("LicenseDependsOn", (value))
}

// GetLicenseDependsOn gets the value of LicenseDependsOn for the instance
func (instance *SoftwareLicensingProduct) GetPropertyLicenseDependsOn() (value string, err error) {
	retValue, err := instance.GetProperty("LicenseDependsOn")
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

// SetLicenseFamily sets the value of LicenseFamily for the instance
func (instance *SoftwareLicensingProduct) SetPropertyLicenseFamily(value string) (err error) {
	return instance.SetProperty("LicenseFamily", (value))
}

// GetLicenseFamily gets the value of LicenseFamily for the instance
func (instance *SoftwareLicensingProduct) GetPropertyLicenseFamily() (value string, err error) {
	retValue, err := instance.GetProperty("LicenseFamily")
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

// SetLicenseIsAddon sets the value of LicenseIsAddon for the instance
func (instance *SoftwareLicensingProduct) SetPropertyLicenseIsAddon(value bool) (err error) {
	return instance.SetProperty("LicenseIsAddon", (value))
}

// GetLicenseIsAddon gets the value of LicenseIsAddon for the instance
func (instance *SoftwareLicensingProduct) GetPropertyLicenseIsAddon() (value bool, err error) {
	retValue, err := instance.GetProperty("LicenseIsAddon")
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

// SetLicenseStatus sets the value of LicenseStatus for the instance
func (instance *SoftwareLicensingProduct) SetPropertyLicenseStatus(value uint32) (err error) {
	return instance.SetProperty("LicenseStatus", (value))
}

// GetLicenseStatus gets the value of LicenseStatus for the instance
func (instance *SoftwareLicensingProduct) GetPropertyLicenseStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("LicenseStatus")
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

// SetLicenseStatusReason sets the value of LicenseStatusReason for the instance
func (instance *SoftwareLicensingProduct) SetPropertyLicenseStatusReason(value uint32) (err error) {
	return instance.SetProperty("LicenseStatusReason", (value))
}

// GetLicenseStatusReason gets the value of LicenseStatusReason for the instance
func (instance *SoftwareLicensingProduct) GetPropertyLicenseStatusReason() (value uint32, err error) {
	retValue, err := instance.GetProperty("LicenseStatusReason")
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

// SetMachineURL sets the value of MachineURL for the instance
func (instance *SoftwareLicensingProduct) SetPropertyMachineURL(value string) (err error) {
	return instance.SetProperty("MachineURL", (value))
}

// GetMachineURL gets the value of MachineURL for the instance
func (instance *SoftwareLicensingProduct) GetPropertyMachineURL() (value string, err error) {
	retValue, err := instance.GetProperty("MachineURL")
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

// SetName sets the value of Name for the instance
func (instance *SoftwareLicensingProduct) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *SoftwareLicensingProduct) GetPropertyName() (value string, err error) {
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

// SetOfflineInstallationId sets the value of OfflineInstallationId for the instance
func (instance *SoftwareLicensingProduct) SetPropertyOfflineInstallationId(value string) (err error) {
	return instance.SetProperty("OfflineInstallationId", (value))
}

// GetOfflineInstallationId gets the value of OfflineInstallationId for the instance
func (instance *SoftwareLicensingProduct) GetPropertyOfflineInstallationId() (value string, err error) {
	retValue, err := instance.GetProperty("OfflineInstallationId")
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

// SetPartialProductKey sets the value of PartialProductKey for the instance
func (instance *SoftwareLicensingProduct) SetPropertyPartialProductKey(value string) (err error) {
	return instance.SetProperty("PartialProductKey", (value))
}

// GetPartialProductKey gets the value of PartialProductKey for the instance
func (instance *SoftwareLicensingProduct) GetPropertyPartialProductKey() (value string, err error) {
	retValue, err := instance.GetProperty("PartialProductKey")
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

// SetProcessorURL sets the value of ProcessorURL for the instance
func (instance *SoftwareLicensingProduct) SetPropertyProcessorURL(value string) (err error) {
	return instance.SetProperty("ProcessorURL", (value))
}

// GetProcessorURL gets the value of ProcessorURL for the instance
func (instance *SoftwareLicensingProduct) GetPropertyProcessorURL() (value string, err error) {
	retValue, err := instance.GetProperty("ProcessorURL")
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

// SetProductKeyChannel sets the value of ProductKeyChannel for the instance
func (instance *SoftwareLicensingProduct) SetPropertyProductKeyChannel(value string) (err error) {
	return instance.SetProperty("ProductKeyChannel", (value))
}

// GetProductKeyChannel gets the value of ProductKeyChannel for the instance
func (instance *SoftwareLicensingProduct) GetPropertyProductKeyChannel() (value string, err error) {
	retValue, err := instance.GetProperty("ProductKeyChannel")
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

// SetProductKeyID sets the value of ProductKeyID for the instance
func (instance *SoftwareLicensingProduct) SetPropertyProductKeyID(value string) (err error) {
	return instance.SetProperty("ProductKeyID", (value))
}

// GetProductKeyID gets the value of ProductKeyID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyProductKeyID() (value string, err error) {
	retValue, err := instance.GetProperty("ProductKeyID")
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

// SetProductKeyID2 sets the value of ProductKeyID2 for the instance
func (instance *SoftwareLicensingProduct) SetPropertyProductKeyID2(value string) (err error) {
	return instance.SetProperty("ProductKeyID2", (value))
}

// GetProductKeyID2 gets the value of ProductKeyID2 for the instance
func (instance *SoftwareLicensingProduct) GetPropertyProductKeyID2() (value string, err error) {
	retValue, err := instance.GetProperty("ProductKeyID2")
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

// SetProductKeyURL sets the value of ProductKeyURL for the instance
func (instance *SoftwareLicensingProduct) SetPropertyProductKeyURL(value string) (err error) {
	return instance.SetProperty("ProductKeyURL", (value))
}

// GetProductKeyURL gets the value of ProductKeyURL for the instance
func (instance *SoftwareLicensingProduct) GetPropertyProductKeyURL() (value string, err error) {
	retValue, err := instance.GetProperty("ProductKeyURL")
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

// SetRemainingAppReArmCount sets the value of RemainingAppReArmCount for the instance
func (instance *SoftwareLicensingProduct) SetPropertyRemainingAppReArmCount(value uint32) (err error) {
	return instance.SetProperty("RemainingAppReArmCount", (value))
}

// GetRemainingAppReArmCount gets the value of RemainingAppReArmCount for the instance
func (instance *SoftwareLicensingProduct) GetPropertyRemainingAppReArmCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemainingAppReArmCount")
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

// SetRemainingSkuReArmCount sets the value of RemainingSkuReArmCount for the instance
func (instance *SoftwareLicensingProduct) SetPropertyRemainingSkuReArmCount(value uint32) (err error) {
	return instance.SetProperty("RemainingSkuReArmCount", (value))
}

// GetRemainingSkuReArmCount gets the value of RemainingSkuReArmCount for the instance
func (instance *SoftwareLicensingProduct) GetPropertyRemainingSkuReArmCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemainingSkuReArmCount")
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
func (instance *SoftwareLicensingProduct) SetPropertyRequiredClientCount(value uint32) (err error) {
	return instance.SetProperty("RequiredClientCount", (value))
}

// GetRequiredClientCount gets the value of RequiredClientCount for the instance
func (instance *SoftwareLicensingProduct) GetPropertyRequiredClientCount() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyTokenActivationAdditionalInfo(value string) (err error) {
	return instance.SetProperty("TokenActivationAdditionalInfo", (value))
}

// GetTokenActivationAdditionalInfo gets the value of TokenActivationAdditionalInfo for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTokenActivationAdditionalInfo() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyTokenActivationCertificateThumbprint(value string) (err error) {
	return instance.SetProperty("TokenActivationCertificateThumbprint", (value))
}

// GetTokenActivationCertificateThumbprint gets the value of TokenActivationCertificateThumbprint for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTokenActivationCertificateThumbprint() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyTokenActivationGrantNumber(value uint32) (err error) {
	return instance.SetProperty("TokenActivationGrantNumber", (value))
}

// GetTokenActivationGrantNumber gets the value of TokenActivationGrantNumber for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTokenActivationGrantNumber() (value uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyTokenActivationILID(value string) (err error) {
	return instance.SetProperty("TokenActivationILID", (value))
}

// GetTokenActivationILID gets the value of TokenActivationILID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTokenActivationILID() (value string, err error) {
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
func (instance *SoftwareLicensingProduct) SetPropertyTokenActivationILVID(value uint32) (err error) {
	return instance.SetProperty("TokenActivationILVID", (value))
}

// GetTokenActivationILVID gets the value of TokenActivationILVID for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTokenActivationILVID() (value uint32, err error) {
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

// SetTrustedTime sets the value of TrustedTime for the instance
func (instance *SoftwareLicensingProduct) SetPropertyTrustedTime(value string) (err error) {
	return instance.SetProperty("TrustedTime", (value))
}

// GetTrustedTime gets the value of TrustedTime for the instance
func (instance *SoftwareLicensingProduct) GetPropertyTrustedTime() (value string, err error) {
	retValue, err := instance.GetProperty("TrustedTime")
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

// SetUseLicenseURL sets the value of UseLicenseURL for the instance
func (instance *SoftwareLicensingProduct) SetPropertyUseLicenseURL(value string) (err error) {
	return instance.SetProperty("UseLicenseURL", (value))
}

// GetUseLicenseURL gets the value of UseLicenseURL for the instance
func (instance *SoftwareLicensingProduct) GetPropertyUseLicenseURL() (value string, err error) {
	retValue, err := instance.GetProperty("UseLicenseURL")
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

// SetValidationURL sets the value of ValidationURL for the instance
func (instance *SoftwareLicensingProduct) SetPropertyValidationURL(value string) (err error) {
	return instance.SetProperty("ValidationURL", (value))
}

// GetValidationURL gets the value of ValidationURL for the instance
func (instance *SoftwareLicensingProduct) GetPropertyValidationURL() (value string, err error) {
	retValue, err := instance.GetProperty("ValidationURL")
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
func (instance *SoftwareLicensingProduct) SetPropertyVLActivationInterval(value uint32) (err error) {
	return instance.SetProperty("VLActivationInterval", (value))
}

// GetVLActivationInterval gets the value of VLActivationInterval for the instance
func (instance *SoftwareLicensingProduct) GetPropertyVLActivationInterval() (value uint32, err error) {
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

// SetVLActivationType sets the value of VLActivationType for the instance
func (instance *SoftwareLicensingProduct) SetPropertyVLActivationType(value uint32) (err error) {
	return instance.SetProperty("VLActivationType", (value))
}

// GetVLActivationType gets the value of VLActivationType for the instance
func (instance *SoftwareLicensingProduct) GetPropertyVLActivationType() (value uint32, err error) {
	retValue, err := instance.GetProperty("VLActivationType")
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

// SetVLActivationTypeEnabled sets the value of VLActivationTypeEnabled for the instance
func (instance *SoftwareLicensingProduct) SetPropertyVLActivationTypeEnabled(value uint32) (err error) {
	return instance.SetProperty("VLActivationTypeEnabled", (value))
}

// GetVLActivationTypeEnabled gets the value of VLActivationTypeEnabled for the instance
func (instance *SoftwareLicensingProduct) GetPropertyVLActivationTypeEnabled() (value uint32, err error) {
	retValue, err := instance.GetProperty("VLActivationTypeEnabled")
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
func (instance *SoftwareLicensingProduct) SetPropertyVLRenewalInterval(value uint32) (err error) {
	return instance.SetProperty("VLRenewalInterval", (value))
}

// GetVLRenewalInterval gets the value of VLRenewalInterval for the instance
func (instance *SoftwareLicensingProduct) GetPropertyVLRenewalInterval() (value uint32, err error) {
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

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) UninstallProductKey() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("UninstallProductKey")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) Activate() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Activate")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ConfirmationId" type="string "></param>
// <param name="InstallationId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) DepositOfflineConfirmationId( /* IN */ InstallationId string,
	/* IN */ ConfirmationId string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DepositOfflineConfirmationId", InstallationId, ConfirmationId)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PolicyName" type="string "></param>

// <param name="PolicyValue" type="uint32 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) GetPolicyInformationDWord( /* IN */ PolicyName string,
	/* OUT */ PolicyValue uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetPolicyInformationDWord", PolicyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PolicyName" type="string "></param>

// <param name="PolicyValue" type="string "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) GetPolicyInformationString( /* IN */ PolicyName string,
	/* OUT */ PolicyValue string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetPolicyInformationString", PolicyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="MachineName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) SetKeyManagementServiceMachine( /* IN */ MachineName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServiceMachine", MachineName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) ClearKeyManagementServiceMachine() (result uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetKeyManagementServicePort( /* IN */ PortNumber uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServicePort", PortNumber)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) ClearKeyManagementServicePort() (result uint32, err error) {
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
func (instance *SoftwareLicensingProduct) SetKeyManagementServiceLookupDomain( /* IN */ LookupDomain string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeyManagementServiceLookupDomain", LookupDomain)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) ClearKeyManagementServiceLookupDomain() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearKeyManagementServiceLookupDomain")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Grants" type="string []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) GetTokenActivationGrants( /* OUT */ Grants []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetTokenActivationGrants")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Challenge" type="string "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) GenerateTokenActivationChallenge( /* OUT */ Challenge string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GenerateTokenActivationChallenge")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="CertChain" type="string "></param>
// <param name="Challenge" type="string "></param>
// <param name="Response" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) DepositTokenActivationResponse( /* IN */ Challenge string,
	/* IN */ Response string,
	/* IN */ CertChain string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DepositTokenActivationResponse", Challenge, Response, CertChain)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ActivationType" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) SetVLActivationTypeEnabled( /* IN */ ActivationType uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVLActivationTypeEnabled", ActivationType)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) ClearVLActivationTypeEnabled() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearVLActivationTypeEnabled")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *SoftwareLicensingProduct) ReArmSku() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ReArmSku")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
