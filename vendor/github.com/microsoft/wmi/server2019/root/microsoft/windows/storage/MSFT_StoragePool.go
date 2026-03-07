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

// MSFT_StoragePool struct
type MSFT_StoragePool struct {
	*MSFT_StorageObject

	//
	AllocatedSize uint64

	//
	ClearOnDeallocate bool

	//
	EnclosureAwareDefault bool

	//
	FaultDomainAwarenessDefault uint16

	//
	FriendlyName string

	//
	HealthStatus uint16

	//
	IsClustered bool

	//
	IsPowerProtected bool

	//
	IsPrimordial bool

	//
	IsReadOnly bool

	//
	LogicalSectorSize uint64

	//
	MediaTypeDefault uint16

	//
	Name string

	//
	OperationalStatus []uint16

	//
	OtherOperationalStatusDescription string

	//
	OtherUsageDescription string

	//
	PhysicalSectorSize uint64

	//
	ProvisioningTypeDefault uint16

	//
	ReadOnlyReason uint16

	//
	RepairPolicy uint16

	//
	ResiliencySettingNameDefault string

	//
	RetireMissingPhysicalDisks uint16

	//
	Size uint64

	//
	SupportedProvisioningTypes []uint16

	//
	SupportsDeduplication bool

	//
	ThinProvisioningAlertThresholds []uint16

	//
	Usage uint16

	//
	Version uint16

	//
	WriteCacheSizeDefault uint64

	//
	WriteCacheSizeMax uint64

	//
	WriteCacheSizeMin uint64
}

func NewMSFT_StoragePoolEx1(instance *cim.WmiInstance) (newInstance *MSFT_StoragePool, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePool{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StoragePoolEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StoragePool, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePool{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAllocatedSize sets the value of AllocatedSize for the instance
func (instance *MSFT_StoragePool) SetPropertyAllocatedSize(value uint64) (err error) {
	return instance.SetProperty("AllocatedSize", (value))
}

// GetAllocatedSize gets the value of AllocatedSize for the instance
func (instance *MSFT_StoragePool) GetPropertyAllocatedSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AllocatedSize")
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

// SetClearOnDeallocate sets the value of ClearOnDeallocate for the instance
func (instance *MSFT_StoragePool) SetPropertyClearOnDeallocate(value bool) (err error) {
	return instance.SetProperty("ClearOnDeallocate", (value))
}

// GetClearOnDeallocate gets the value of ClearOnDeallocate for the instance
func (instance *MSFT_StoragePool) GetPropertyClearOnDeallocate() (value bool, err error) {
	retValue, err := instance.GetProperty("ClearOnDeallocate")
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

// SetEnclosureAwareDefault sets the value of EnclosureAwareDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyEnclosureAwareDefault(value bool) (err error) {
	return instance.SetProperty("EnclosureAwareDefault", (value))
}

// GetEnclosureAwareDefault gets the value of EnclosureAwareDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyEnclosureAwareDefault() (value bool, err error) {
	retValue, err := instance.GetProperty("EnclosureAwareDefault")
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

// SetFaultDomainAwarenessDefault sets the value of FaultDomainAwarenessDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyFaultDomainAwarenessDefault(value uint16) (err error) {
	return instance.SetProperty("FaultDomainAwarenessDefault", (value))
}

// GetFaultDomainAwarenessDefault gets the value of FaultDomainAwarenessDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyFaultDomainAwarenessDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("FaultDomainAwarenessDefault")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *MSFT_StoragePool) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_StoragePool) GetPropertyFriendlyName() (value string, err error) {
	retValue, err := instance.GetProperty("FriendlyName")
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_StoragePool) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_StoragePool) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetIsClustered sets the value of IsClustered for the instance
func (instance *MSFT_StoragePool) SetPropertyIsClustered(value bool) (err error) {
	return instance.SetProperty("IsClustered", (value))
}

// GetIsClustered gets the value of IsClustered for the instance
func (instance *MSFT_StoragePool) GetPropertyIsClustered() (value bool, err error) {
	retValue, err := instance.GetProperty("IsClustered")
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

// SetIsPowerProtected sets the value of IsPowerProtected for the instance
func (instance *MSFT_StoragePool) SetPropertyIsPowerProtected(value bool) (err error) {
	return instance.SetProperty("IsPowerProtected", (value))
}

// GetIsPowerProtected gets the value of IsPowerProtected for the instance
func (instance *MSFT_StoragePool) GetPropertyIsPowerProtected() (value bool, err error) {
	retValue, err := instance.GetProperty("IsPowerProtected")
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

// SetIsPrimordial sets the value of IsPrimordial for the instance
func (instance *MSFT_StoragePool) SetPropertyIsPrimordial(value bool) (err error) {
	return instance.SetProperty("IsPrimordial", (value))
}

// GetIsPrimordial gets the value of IsPrimordial for the instance
func (instance *MSFT_StoragePool) GetPropertyIsPrimordial() (value bool, err error) {
	retValue, err := instance.GetProperty("IsPrimordial")
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

// SetIsReadOnly sets the value of IsReadOnly for the instance
func (instance *MSFT_StoragePool) SetPropertyIsReadOnly(value bool) (err error) {
	return instance.SetProperty("IsReadOnly", (value))
}

// GetIsReadOnly gets the value of IsReadOnly for the instance
func (instance *MSFT_StoragePool) GetPropertyIsReadOnly() (value bool, err error) {
	retValue, err := instance.GetProperty("IsReadOnly")
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

// SetLogicalSectorSize sets the value of LogicalSectorSize for the instance
func (instance *MSFT_StoragePool) SetPropertyLogicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("LogicalSectorSize", (value))
}

// GetLogicalSectorSize gets the value of LogicalSectorSize for the instance
func (instance *MSFT_StoragePool) GetPropertyLogicalSectorSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("LogicalSectorSize")
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

// SetMediaTypeDefault sets the value of MediaTypeDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyMediaTypeDefault(value uint16) (err error) {
	return instance.SetProperty("MediaTypeDefault", (value))
}

// GetMediaTypeDefault gets the value of MediaTypeDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyMediaTypeDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("MediaTypeDefault")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetName sets the value of Name for the instance
func (instance *MSFT_StoragePool) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_StoragePool) GetPropertyName() (value string, err error) {
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_StoragePool) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_StoragePool) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetOtherOperationalStatusDescription sets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_StoragePool) SetPropertyOtherOperationalStatusDescription(value string) (err error) {
	return instance.SetProperty("OtherOperationalStatusDescription", (value))
}

// GetOtherOperationalStatusDescription gets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_StoragePool) GetPropertyOtherOperationalStatusDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherOperationalStatusDescription")
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

// SetOtherUsageDescription sets the value of OtherUsageDescription for the instance
func (instance *MSFT_StoragePool) SetPropertyOtherUsageDescription(value string) (err error) {
	return instance.SetProperty("OtherUsageDescription", (value))
}

// GetOtherUsageDescription gets the value of OtherUsageDescription for the instance
func (instance *MSFT_StoragePool) GetPropertyOtherUsageDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherUsageDescription")
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

// SetPhysicalSectorSize sets the value of PhysicalSectorSize for the instance
func (instance *MSFT_StoragePool) SetPropertyPhysicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("PhysicalSectorSize", (value))
}

// GetPhysicalSectorSize gets the value of PhysicalSectorSize for the instance
func (instance *MSFT_StoragePool) GetPropertyPhysicalSectorSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("PhysicalSectorSize")
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

// SetProvisioningTypeDefault sets the value of ProvisioningTypeDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyProvisioningTypeDefault(value uint16) (err error) {
	return instance.SetProperty("ProvisioningTypeDefault", (value))
}

// GetProvisioningTypeDefault gets the value of ProvisioningTypeDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyProvisioningTypeDefault() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProvisioningTypeDefault")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetReadOnlyReason sets the value of ReadOnlyReason for the instance
func (instance *MSFT_StoragePool) SetPropertyReadOnlyReason(value uint16) (err error) {
	return instance.SetProperty("ReadOnlyReason", (value))
}

// GetReadOnlyReason gets the value of ReadOnlyReason for the instance
func (instance *MSFT_StoragePool) GetPropertyReadOnlyReason() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReadOnlyReason")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetRepairPolicy sets the value of RepairPolicy for the instance
func (instance *MSFT_StoragePool) SetPropertyRepairPolicy(value uint16) (err error) {
	return instance.SetProperty("RepairPolicy", (value))
}

// GetRepairPolicy gets the value of RepairPolicy for the instance
func (instance *MSFT_StoragePool) GetPropertyRepairPolicy() (value uint16, err error) {
	retValue, err := instance.GetProperty("RepairPolicy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetResiliencySettingNameDefault sets the value of ResiliencySettingNameDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyResiliencySettingNameDefault(value string) (err error) {
	return instance.SetProperty("ResiliencySettingNameDefault", (value))
}

// GetResiliencySettingNameDefault gets the value of ResiliencySettingNameDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyResiliencySettingNameDefault() (value string, err error) {
	retValue, err := instance.GetProperty("ResiliencySettingNameDefault")
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

// SetRetireMissingPhysicalDisks sets the value of RetireMissingPhysicalDisks for the instance
func (instance *MSFT_StoragePool) SetPropertyRetireMissingPhysicalDisks(value uint16) (err error) {
	return instance.SetProperty("RetireMissingPhysicalDisks", (value))
}

// GetRetireMissingPhysicalDisks gets the value of RetireMissingPhysicalDisks for the instance
func (instance *MSFT_StoragePool) GetPropertyRetireMissingPhysicalDisks() (value uint16, err error) {
	retValue, err := instance.GetProperty("RetireMissingPhysicalDisks")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetSize sets the value of Size for the instance
func (instance *MSFT_StoragePool) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_StoragePool) GetPropertySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("Size")
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

// SetSupportedProvisioningTypes sets the value of SupportedProvisioningTypes for the instance
func (instance *MSFT_StoragePool) SetPropertySupportedProvisioningTypes(value []uint16) (err error) {
	return instance.SetProperty("SupportedProvisioningTypes", (value))
}

// GetSupportedProvisioningTypes gets the value of SupportedProvisioningTypes for the instance
func (instance *MSFT_StoragePool) GetPropertySupportedProvisioningTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedProvisioningTypes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportsDeduplication sets the value of SupportsDeduplication for the instance
func (instance *MSFT_StoragePool) SetPropertySupportsDeduplication(value bool) (err error) {
	return instance.SetProperty("SupportsDeduplication", (value))
}

// GetSupportsDeduplication gets the value of SupportsDeduplication for the instance
func (instance *MSFT_StoragePool) GetPropertySupportsDeduplication() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsDeduplication")
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

// SetThinProvisioningAlertThresholds sets the value of ThinProvisioningAlertThresholds for the instance
func (instance *MSFT_StoragePool) SetPropertyThinProvisioningAlertThresholds(value []uint16) (err error) {
	return instance.SetProperty("ThinProvisioningAlertThresholds", (value))
}

// GetThinProvisioningAlertThresholds gets the value of ThinProvisioningAlertThresholds for the instance
func (instance *MSFT_StoragePool) GetPropertyThinProvisioningAlertThresholds() (value []uint16, err error) {
	retValue, err := instance.GetProperty("ThinProvisioningAlertThresholds")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetUsage sets the value of Usage for the instance
func (instance *MSFT_StoragePool) SetPropertyUsage(value uint16) (err error) {
	return instance.SetProperty("Usage", (value))
}

// GetUsage gets the value of Usage for the instance
func (instance *MSFT_StoragePool) GetPropertyUsage() (value uint16, err error) {
	retValue, err := instance.GetProperty("Usage")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetVersion sets the value of Version for the instance
func (instance *MSFT_StoragePool) SetPropertyVersion(value uint16) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *MSFT_StoragePool) GetPropertyVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("Version")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetWriteCacheSizeDefault sets the value of WriteCacheSizeDefault for the instance
func (instance *MSFT_StoragePool) SetPropertyWriteCacheSizeDefault(value uint64) (err error) {
	return instance.SetProperty("WriteCacheSizeDefault", (value))
}

// GetWriteCacheSizeDefault gets the value of WriteCacheSizeDefault for the instance
func (instance *MSFT_StoragePool) GetPropertyWriteCacheSizeDefault() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheSizeDefault")
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

// SetWriteCacheSizeMax sets the value of WriteCacheSizeMax for the instance
func (instance *MSFT_StoragePool) SetPropertyWriteCacheSizeMax(value uint64) (err error) {
	return instance.SetProperty("WriteCacheSizeMax", (value))
}

// GetWriteCacheSizeMax gets the value of WriteCacheSizeMax for the instance
func (instance *MSFT_StoragePool) GetPropertyWriteCacheSizeMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheSizeMax")
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

// SetWriteCacheSizeMin sets the value of WriteCacheSizeMin for the instance
func (instance *MSFT_StoragePool) SetPropertyWriteCacheSizeMin(value uint64) (err error) {
	return instance.SetProperty("WriteCacheSizeMin", (value))
}

// GetWriteCacheSizeMin gets the value of WriteCacheSizeMin for the instance
func (instance *MSFT_StoragePool) GetPropertyWriteCacheSizeMin() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheSizeMin")
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

//

// <param name="AllocationUnitSize" type="uint64 "></param>
// <param name="AutoNumberOfColumns" type="bool "></param>
// <param name="AutoWriteCacheSize" type="bool "></param>
// <param name="ColumnIsolation" type="uint16 "></param>
// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="IsEnclosureAware" type="bool "></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="NumberOfGroups" type="uint16 "></param>
// <param name="OtherUsageDescription" type="string "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="PhysicalDisksToUse" type="MSFT_PhysicalDisk []"></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ReadCacheSize" type="uint64 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>
// <param name="StorageFaultDomainsToUse" type="MSFT_StorageFaultDomain []"></param>
// <param name="StorageTiers" type="MSFT_StorageTier []"></param>
// <param name="StorageTierSizes" type="uint64 []"></param>
// <param name="Usage" type="uint16 "></param>
// <param name="UseMaximumSize" type="bool "></param>
// <param name="WriteCacheSize" type="uint64 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) CreateVirtualDisk( /* IN */ FriendlyName string,
	/* IN */ Size uint64,
	/* IN */ UseMaximumSize bool,
	/* IN */ ProvisioningType uint16,
	/* IN */ AllocationUnitSize uint64,
	/* IN */ MediaType uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ AutoNumberOfColumns bool,
	/* IN */ Interleave uint64,
	/* IN */ NumberOfGroups uint16,
	/* IN */ IsEnclosureAware bool,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ColumnIsolation uint16,
	/* IN */ PhysicalDisksToUse []MSFT_PhysicalDisk,
	/* IN */ StorageFaultDomainsToUse []MSFT_StorageFaultDomain,
	/* IN */ StorageTiers []MSFT_StorageTier,
	/* IN */ StorageTierSizes []uint64,
	/* IN */ WriteCacheSize uint64,
	/* IN */ AutoWriteCacheSize bool,
	/* IN */ ReadCacheSize uint64,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedVirtualDisk MSFT_VirtualDisk,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateVirtualDisk", FriendlyName, Size, UseMaximumSize, ProvisioningType, AllocationUnitSize, MediaType, ResiliencySettingName, Usage, OtherUsageDescription, NumberOfDataCopies, PhysicalDiskRedundancy, NumberOfColumns, AutoNumberOfColumns, Interleave, NumberOfGroups, IsEnclosureAware, FaultDomainAwareness, ColumnIsolation, PhysicalDisksToUse, StorageFaultDomainsToUse, StorageTiers, StorageTierSizes, WriteCacheSize, AutoWriteCacheSize, ReadCacheSize, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AccessPath" type="string "></param>
// <param name="AllocationUnitSize" type="uint32 "></param>
// <param name="FileServer" type="MSFT_FileServer "></param>
// <param name="FileSystem" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ReadCacheSize" type="uint64 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>
// <param name="StorageTiers" type="MSFT_StorageTier []"></param>
// <param name="StorageTierSizes" type="uint64 []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVolume" type="MSFT_Volume "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) CreateVolume( /* IN */ FriendlyName string,
	/* IN */ Size uint64,
	/* IN */ StorageTiers []MSFT_StorageTier,
	/* IN */ StorageTierSizes []uint64,
	/* IN */ ProvisioningType uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ FileSystem uint16,
	/* IN */ AccessPath string,
	/* IN */ AllocationUnitSize uint32,
	/* IN */ ReadCacheSize uint64,
	/* IN */ FileServer MSFT_FileServer,
	/* OUT */ CreatedVolume MSFT_Volume,
	/* OPTIONAL IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateVolume", FriendlyName, Size, StorageTiers, StorageTierSizes, ProvisioningType, ResiliencySettingName, PhysicalDiskRedundancy, NumberOfColumns, FileSystem, AccessPath, AllocationUnitSize, ReadCacheSize, FileServer, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AllocationUnitSize" type="uint64 "></param>
// <param name="ColumnIsolation" type="uint16 "></param>
// <param name="Description" type="string "></param>
// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="NumberOfGroups" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="StorageFaultDomainsToUse" type="MSFT_StorageFaultDomain []"></param>
// <param name="Usage" type="uint16 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedStorageTier" type="MSFT_StorageTier "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) CreateStorageTier( /* IN */ FriendlyName string,
	/* IN */ ProvisioningType uint16,
	/* IN */ AllocationUnitSize uint64,
	/* IN */ MediaType uint16,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ColumnIsolation uint16,
	/* IN */ StorageFaultDomainsToUse []MSFT_StorageFaultDomain,
	/* IN */ ResiliencySettingName string,
	/* IN */ Usage uint16,
	/* IN */ Interleave uint64,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ NumberOfGroups uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ Description string,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageTier MSFT_StorageTier,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateStorageTier", FriendlyName, ProvisioningType, AllocationUnitSize, MediaType, FaultDomainAwareness, ColumnIsolation, StorageFaultDomainsToUse, ResiliencySettingName, Usage, Interleave, NumberOfDataCopies, NumberOfGroups, NumberOfColumns, PhysicalDiskRedundancy, Description, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) DeleteObject( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) Upgrade( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Upgrade")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) Optimize( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Optimize", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) AddPhysicalDisk( /* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ Usage uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("AddPhysicalDisk", PhysicalDisks, Usage, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) RemovePhysicalDisk( /* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemovePhysicalDisk", PhysicalDisks, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="ResiliencySettingName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedSizes" type="uint64 []"></param>
// <param name="VirtualDiskSizeDivisor" type="uint64 "></param>
// <param name="VirtualDiskSizeMax" type="uint64 "></param>
// <param name="VirtualDiskSizeMin" type="uint64 "></param>
func (instance *MSFT_StoragePool) GetSupportedSize( /* IN */ ResiliencySettingName string,
	/* IN */ FaultDomainAwareness uint16,
	/* OUT */ SupportedSizes []uint64,
	/* OUT */ VirtualDiskSizeMin uint64,
	/* OUT */ VirtualDiskSizeMax uint64,
	/* OUT */ VirtualDiskSizeDivisor uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedSize", ResiliencySettingName, FaultDomainAwareness)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SecurityDescriptor" type="string "></param>
func (instance *MSFT_StoragePool) GetSecurityDescriptor( /* OUT */ SecurityDescriptor string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSecurityDescriptor")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="SecurityDescriptor" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) SetSecurityDescriptor( /* IN */ SecurityDescriptor string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetSecurityDescriptor", SecurityDescriptor)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) SetFriendlyName( /* IN */ FriendlyName string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetFriendlyName", FriendlyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="OtherUsageDescription" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) SetUsage( /* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetUsage", Usage, OtherUsageDescription)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AutoWriteCacheSize" type="bool "></param>
// <param name="EnclosureAwareDefault" type="bool "></param>
// <param name="FaultDomainAwarenessDefault" type="uint16 "></param>
// <param name="MediaTypeDefault" type="uint16 "></param>
// <param name="ProvisioningTypeDefault" type="uint16 "></param>
// <param name="ResiliencySettingNameDefault" type="string "></param>
// <param name="WriteCacheSizeDefault" type="uint64 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) SetDefaults( /* IN */ ProvisioningTypeDefault uint16,
	/* IN */ MediaTypeDefault uint16,
	/* IN */ ResiliencySettingNameDefault string,
	/* IN */ EnclosureAwareDefault bool,
	/* IN */ FaultDomainAwarenessDefault uint16,
	/* IN */ WriteCacheSizeDefault uint64,
	/* IN */ AutoWriteCacheSize bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetDefaults", ProvisioningTypeDefault, MediaTypeDefault, ResiliencySettingNameDefault, EnclosureAwareDefault, FaultDomainAwarenessDefault, WriteCacheSizeDefault, AutoWriteCacheSize)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ClearOnDeallocate" type="bool "></param>
// <param name="IsPowerProtected" type="bool "></param>
// <param name="IsReadOnly" type="bool "></param>
// <param name="RepairPolicy" type="uint16 "></param>
// <param name="RetireMissingPhysicalDisks" type="uint16 "></param>
// <param name="ThinProvisioningAlertThresholds" type="uint16 []"></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StoragePool) SetAttributes( /* IN */ IsReadOnly bool,
	/* IN */ ClearOnDeallocate bool,
	/* IN */ IsPowerProtected bool,
	/* IN */ RepairPolicy uint16,
	/* IN */ RetireMissingPhysicalDisks uint16,
	/* IN */ ThinProvisioningAlertThresholds []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", IsReadOnly, ClearOnDeallocate, IsPowerProtected, RepairPolicy, RetireMissingPhysicalDisks, ThinProvisioningAlertThresholds)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
