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

// MSFT_StorageSubSystem struct
type MSFT_StorageSubSystem struct {
	*MSFT_StorageObject

	//
	AutomaticClusteringEnabled bool

	//
	CimServerName string

	//
	CurrentCacheLevel uint16

	//
	DataTieringType uint16

	//
	Description string

	//
	FaultDomainAwarenessDefault uint16

	//
	FirmwareVersion string

	//
	FriendlyName string

	//
	HealthStatus uint16

	//
	iSCSITargetCreationScheme uint16

	//
	Manufacturer string

	//
	MaskingClientSelectableDeviceNumbers bool

	//
	MaskingMapCountMax uint16

	//
	MaskingOneInitiatorIdPerView bool

	//
	MaskingOtherValidInitiatorIdTypes []string

	//
	MaskingPortsPerView uint16

	//
	MaskingValidInitiatorIdTypes []uint16

	//
	Model string

	//
	Name string

	//
	NameFormat uint16

	//
	NumberOfSlots uint32

	//
	OperationalStatus []uint16

	//
	OtherHostTypeDescription []string

	//
	OtherIdentifyingInfo []string

	//
	OtherIdentifyingInfoDescription []string

	//
	OtherOperationalStatusDescription string

	//
	PhysicalDisksPerStoragePoolMin uint16

	//
	ReplicasPerSourceCloneMax uint16

	//
	ReplicasPerSourceMirrorMax uint16

	//
	ReplicasPerSourceSnapshotMax uint16

	//
	SerialNumber string

	//
	StorageConnectionType uint16

	//
	SupportedDeduplicationFileSystemTypes []uint16

	//
	SupportedDeduplicationObjectTypes []uint16

	//
	SupportedFileServerProtocols []uint16

	//
	SupportedFileSystems []uint16

	//
	SupportedHostType []uint16

	//
	SupportsAutomaticStoragePoolSelection bool

	//
	SupportsCloneLocal bool

	//
	SupportsCloneRemote bool

	//
	SupportsContinuouslyAvailableFileServer bool

	//
	SupportsFileServer bool

	//
	SupportsFileServerCreation bool

	//
	SupportsMaskingVirtualDiskToHosts bool

	//
	SupportsMirrorLocal bool

	//
	SupportsMirrorRemote bool

	//
	SupportsMultipleResiliencySettingsPerStoragePool bool

	//
	SupportsSnapshotLocal bool

	//
	SupportsSnapshotRemote bool

	//
	SupportsStoragePoolAddPhysicalDisk bool

	//
	SupportsStoragePoolCreation bool

	//
	SupportsStoragePoolDeletion bool

	//
	SupportsStoragePoolFriendlyNameModification bool

	//
	SupportsStoragePoolRemovePhysicalDisk bool

	//
	SupportsStorageTierCreation bool

	//
	SupportsStorageTierDeletion bool

	//
	SupportsStorageTieredVirtualDiskCreation bool

	//
	SupportsStorageTierFriendlyNameModification bool

	//
	SupportsStorageTierResize bool

	//
	SupportsVirtualDiskCapacityExpansion bool

	//
	SupportsVirtualDiskCapacityReduction bool

	//
	SupportsVirtualDiskCreation bool

	//
	SupportsVirtualDiskDeletion bool

	//
	SupportsVirtualDiskModification bool

	//
	SupportsVirtualDiskRepair bool

	//
	SupportsVolumeCreation bool

	//
	Tag string
}

func NewMSFT_StorageSubSystemEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystem, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystem{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystem, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystem{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAutomaticClusteringEnabled sets the value of AutomaticClusteringEnabled for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyAutomaticClusteringEnabled(value bool) (err error) {
	return instance.SetProperty("AutomaticClusteringEnabled", (value))
}

// GetAutomaticClusteringEnabled gets the value of AutomaticClusteringEnabled for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyAutomaticClusteringEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("AutomaticClusteringEnabled")
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

// SetCimServerName sets the value of CimServerName for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyCimServerName(value string) (err error) {
	return instance.SetProperty("CimServerName", (value))
}

// GetCimServerName gets the value of CimServerName for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyCimServerName() (value string, err error) {
	retValue, err := instance.GetProperty("CimServerName")
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

// SetCurrentCacheLevel sets the value of CurrentCacheLevel for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyCurrentCacheLevel(value uint16) (err error) {
	return instance.SetProperty("CurrentCacheLevel", (value))
}

// GetCurrentCacheLevel gets the value of CurrentCacheLevel for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyCurrentCacheLevel() (value uint16, err error) {
	retValue, err := instance.GetProperty("CurrentCacheLevel")
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

// SetDataTieringType sets the value of DataTieringType for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyDataTieringType(value uint16) (err error) {
	return instance.SetProperty("DataTieringType", (value))
}

// GetDataTieringType gets the value of DataTieringType for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyDataTieringType() (value uint16, err error) {
	retValue, err := instance.GetProperty("DataTieringType")
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

// SetDescription sets the value of Description for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyDescription() (value string, err error) {
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

// SetFaultDomainAwarenessDefault sets the value of FaultDomainAwarenessDefault for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyFaultDomainAwarenessDefault(value uint16) (err error) {
	return instance.SetProperty("FaultDomainAwarenessDefault", (value))
}

// GetFaultDomainAwarenessDefault gets the value of FaultDomainAwarenessDefault for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyFaultDomainAwarenessDefault() (value uint16, err error) {
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

// SetFirmwareVersion sets the value of FirmwareVersion for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyFirmwareVersion(value string) (err error) {
	return instance.SetProperty("FirmwareVersion", (value))
}

// GetFirmwareVersion gets the value of FirmwareVersion for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyFirmwareVersion() (value string, err error) {
	retValue, err := instance.GetProperty("FirmwareVersion")
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

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyFriendlyName() (value string, err error) {
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
func (instance *MSFT_StorageSubSystem) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyHealthStatus() (value uint16, err error) {
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

// SetiSCSITargetCreationScheme sets the value of iSCSITargetCreationScheme for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyiSCSITargetCreationScheme(value uint16) (err error) {
	return instance.SetProperty("iSCSITargetCreationScheme", (value))
}

// GetiSCSITargetCreationScheme gets the value of iSCSITargetCreationScheme for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyiSCSITargetCreationScheme() (value uint16, err error) {
	retValue, err := instance.GetProperty("iSCSITargetCreationScheme")
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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

// SetMaskingClientSelectableDeviceNumbers sets the value of MaskingClientSelectableDeviceNumbers for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingClientSelectableDeviceNumbers(value bool) (err error) {
	return instance.SetProperty("MaskingClientSelectableDeviceNumbers", (value))
}

// GetMaskingClientSelectableDeviceNumbers gets the value of MaskingClientSelectableDeviceNumbers for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingClientSelectableDeviceNumbers() (value bool, err error) {
	retValue, err := instance.GetProperty("MaskingClientSelectableDeviceNumbers")
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

// SetMaskingMapCountMax sets the value of MaskingMapCountMax for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingMapCountMax(value uint16) (err error) {
	return instance.SetProperty("MaskingMapCountMax", (value))
}

// GetMaskingMapCountMax gets the value of MaskingMapCountMax for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingMapCountMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("MaskingMapCountMax")
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

// SetMaskingOneInitiatorIdPerView sets the value of MaskingOneInitiatorIdPerView for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingOneInitiatorIdPerView(value bool) (err error) {
	return instance.SetProperty("MaskingOneInitiatorIdPerView", (value))
}

// GetMaskingOneInitiatorIdPerView gets the value of MaskingOneInitiatorIdPerView for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingOneInitiatorIdPerView() (value bool, err error) {
	retValue, err := instance.GetProperty("MaskingOneInitiatorIdPerView")
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

// SetMaskingOtherValidInitiatorIdTypes sets the value of MaskingOtherValidInitiatorIdTypes for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingOtherValidInitiatorIdTypes(value []string) (err error) {
	return instance.SetProperty("MaskingOtherValidInitiatorIdTypes", (value))
}

// GetMaskingOtherValidInitiatorIdTypes gets the value of MaskingOtherValidInitiatorIdTypes for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingOtherValidInitiatorIdTypes() (value []string, err error) {
	retValue, err := instance.GetProperty("MaskingOtherValidInitiatorIdTypes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetMaskingPortsPerView sets the value of MaskingPortsPerView for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingPortsPerView(value uint16) (err error) {
	return instance.SetProperty("MaskingPortsPerView", (value))
}

// GetMaskingPortsPerView gets the value of MaskingPortsPerView for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingPortsPerView() (value uint16, err error) {
	retValue, err := instance.GetProperty("MaskingPortsPerView")
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

// SetMaskingValidInitiatorIdTypes sets the value of MaskingValidInitiatorIdTypes for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyMaskingValidInitiatorIdTypes(value []uint16) (err error) {
	return instance.SetProperty("MaskingValidInitiatorIdTypes", (value))
}

// GetMaskingValidInitiatorIdTypes gets the value of MaskingValidInitiatorIdTypes for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyMaskingValidInitiatorIdTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("MaskingValidInitiatorIdTypes")
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

// SetModel sets the value of Model for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyModel(value string) (err error) {
	return instance.SetProperty("Model", (value))
}

// GetModel gets the value of Model for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyModel() (value string, err error) {
	retValue, err := instance.GetProperty("Model")
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
func (instance *MSFT_StorageSubSystem) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyName() (value string, err error) {
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

// SetNameFormat sets the value of NameFormat for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyNameFormat(value uint16) (err error) {
	return instance.SetProperty("NameFormat", (value))
}

// GetNameFormat gets the value of NameFormat for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyNameFormat() (value uint16, err error) {
	retValue, err := instance.GetProperty("NameFormat")
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

// SetNumberOfSlots sets the value of NumberOfSlots for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyNumberOfSlots(value uint32) (err error) {
	return instance.SetProperty("NumberOfSlots", (value))
}

// GetNumberOfSlots gets the value of NumberOfSlots for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyNumberOfSlots() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfSlots")
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyOperationalStatus() (value []uint16, err error) {
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

// SetOtherHostTypeDescription sets the value of OtherHostTypeDescription for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyOtherHostTypeDescription(value []string) (err error) {
	return instance.SetProperty("OtherHostTypeDescription", (value))
}

// GetOtherHostTypeDescription gets the value of OtherHostTypeDescription for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyOtherHostTypeDescription() (value []string, err error) {
	retValue, err := instance.GetProperty("OtherHostTypeDescription")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetOtherIdentifyingInfo sets the value of OtherIdentifyingInfo for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyOtherIdentifyingInfo(value []string) (err error) {
	return instance.SetProperty("OtherIdentifyingInfo", (value))
}

// GetOtherIdentifyingInfo gets the value of OtherIdentifyingInfo for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyOtherIdentifyingInfo() (value []string, err error) {
	retValue, err := instance.GetProperty("OtherIdentifyingInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetOtherIdentifyingInfoDescription sets the value of OtherIdentifyingInfoDescription for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyOtherIdentifyingInfoDescription(value []string) (err error) {
	return instance.SetProperty("OtherIdentifyingInfoDescription", (value))
}

// GetOtherIdentifyingInfoDescription gets the value of OtherIdentifyingInfoDescription for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyOtherIdentifyingInfoDescription() (value []string, err error) {
	retValue, err := instance.GetProperty("OtherIdentifyingInfoDescription")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetOtherOperationalStatusDescription sets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyOtherOperationalStatusDescription(value string) (err error) {
	return instance.SetProperty("OtherOperationalStatusDescription", (value))
}

// GetOtherOperationalStatusDescription gets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyOtherOperationalStatusDescription() (value string, err error) {
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

// SetPhysicalDisksPerStoragePoolMin sets the value of PhysicalDisksPerStoragePoolMin for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyPhysicalDisksPerStoragePoolMin(value uint16) (err error) {
	return instance.SetProperty("PhysicalDisksPerStoragePoolMin", (value))
}

// GetPhysicalDisksPerStoragePoolMin gets the value of PhysicalDisksPerStoragePoolMin for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyPhysicalDisksPerStoragePoolMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("PhysicalDisksPerStoragePoolMin")
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

// SetReplicasPerSourceCloneMax sets the value of ReplicasPerSourceCloneMax for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyReplicasPerSourceCloneMax(value uint16) (err error) {
	return instance.SetProperty("ReplicasPerSourceCloneMax", (value))
}

// GetReplicasPerSourceCloneMax gets the value of ReplicasPerSourceCloneMax for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyReplicasPerSourceCloneMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReplicasPerSourceCloneMax")
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

// SetReplicasPerSourceMirrorMax sets the value of ReplicasPerSourceMirrorMax for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyReplicasPerSourceMirrorMax(value uint16) (err error) {
	return instance.SetProperty("ReplicasPerSourceMirrorMax", (value))
}

// GetReplicasPerSourceMirrorMax gets the value of ReplicasPerSourceMirrorMax for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyReplicasPerSourceMirrorMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReplicasPerSourceMirrorMax")
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

// SetReplicasPerSourceSnapshotMax sets the value of ReplicasPerSourceSnapshotMax for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyReplicasPerSourceSnapshotMax(value uint16) (err error) {
	return instance.SetProperty("ReplicasPerSourceSnapshotMax", (value))
}

// GetReplicasPerSourceSnapshotMax gets the value of ReplicasPerSourceSnapshotMax for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyReplicasPerSourceSnapshotMax() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReplicasPerSourceSnapshotMax")
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

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("SerialNumber")
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

// SetStorageConnectionType sets the value of StorageConnectionType for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyStorageConnectionType(value uint16) (err error) {
	return instance.SetProperty("StorageConnectionType", (value))
}

// GetStorageConnectionType gets the value of StorageConnectionType for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyStorageConnectionType() (value uint16, err error) {
	retValue, err := instance.GetProperty("StorageConnectionType")
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

// SetSupportedDeduplicationFileSystemTypes sets the value of SupportedDeduplicationFileSystemTypes for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportedDeduplicationFileSystemTypes(value []uint16) (err error) {
	return instance.SetProperty("SupportedDeduplicationFileSystemTypes", (value))
}

// GetSupportedDeduplicationFileSystemTypes gets the value of SupportedDeduplicationFileSystemTypes for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportedDeduplicationFileSystemTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedDeduplicationFileSystemTypes")
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

// SetSupportedDeduplicationObjectTypes sets the value of SupportedDeduplicationObjectTypes for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportedDeduplicationObjectTypes(value []uint16) (err error) {
	return instance.SetProperty("SupportedDeduplicationObjectTypes", (value))
}

// GetSupportedDeduplicationObjectTypes gets the value of SupportedDeduplicationObjectTypes for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportedDeduplicationObjectTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedDeduplicationObjectTypes")
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

// SetSupportedFileServerProtocols sets the value of SupportedFileServerProtocols for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportedFileServerProtocols(value []uint16) (err error) {
	return instance.SetProperty("SupportedFileServerProtocols", (value))
}

// GetSupportedFileServerProtocols gets the value of SupportedFileServerProtocols for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportedFileServerProtocols() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedFileServerProtocols")
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

// SetSupportedFileSystems sets the value of SupportedFileSystems for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportedFileSystems(value []uint16) (err error) {
	return instance.SetProperty("SupportedFileSystems", (value))
}

// GetSupportedFileSystems gets the value of SupportedFileSystems for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportedFileSystems() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedFileSystems")
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

// SetSupportedHostType sets the value of SupportedHostType for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportedHostType(value []uint16) (err error) {
	return instance.SetProperty("SupportedHostType", (value))
}

// GetSupportedHostType gets the value of SupportedHostType for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportedHostType() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedHostType")
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

// SetSupportsAutomaticStoragePoolSelection sets the value of SupportsAutomaticStoragePoolSelection for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsAutomaticStoragePoolSelection(value bool) (err error) {
	return instance.SetProperty("SupportsAutomaticStoragePoolSelection", (value))
}

// GetSupportsAutomaticStoragePoolSelection gets the value of SupportsAutomaticStoragePoolSelection for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsAutomaticStoragePoolSelection() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsAutomaticStoragePoolSelection")
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

// SetSupportsCloneLocal sets the value of SupportsCloneLocal for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsCloneLocal(value bool) (err error) {
	return instance.SetProperty("SupportsCloneLocal", (value))
}

// GetSupportsCloneLocal gets the value of SupportsCloneLocal for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsCloneLocal() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsCloneLocal")
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

// SetSupportsCloneRemote sets the value of SupportsCloneRemote for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsCloneRemote(value bool) (err error) {
	return instance.SetProperty("SupportsCloneRemote", (value))
}

// GetSupportsCloneRemote gets the value of SupportsCloneRemote for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsCloneRemote() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsCloneRemote")
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

// SetSupportsContinuouslyAvailableFileServer sets the value of SupportsContinuouslyAvailableFileServer for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsContinuouslyAvailableFileServer(value bool) (err error) {
	return instance.SetProperty("SupportsContinuouslyAvailableFileServer", (value))
}

// GetSupportsContinuouslyAvailableFileServer gets the value of SupportsContinuouslyAvailableFileServer for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsContinuouslyAvailableFileServer() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsContinuouslyAvailableFileServer")
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

// SetSupportsFileServer sets the value of SupportsFileServer for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsFileServer(value bool) (err error) {
	return instance.SetProperty("SupportsFileServer", (value))
}

// GetSupportsFileServer gets the value of SupportsFileServer for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsFileServer() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsFileServer")
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

// SetSupportsFileServerCreation sets the value of SupportsFileServerCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsFileServerCreation(value bool) (err error) {
	return instance.SetProperty("SupportsFileServerCreation", (value))
}

// GetSupportsFileServerCreation gets the value of SupportsFileServerCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsFileServerCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsFileServerCreation")
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

// SetSupportsMaskingVirtualDiskToHosts sets the value of SupportsMaskingVirtualDiskToHosts for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsMaskingVirtualDiskToHosts(value bool) (err error) {
	return instance.SetProperty("SupportsMaskingVirtualDiskToHosts", (value))
}

// GetSupportsMaskingVirtualDiskToHosts gets the value of SupportsMaskingVirtualDiskToHosts for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsMaskingVirtualDiskToHosts() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsMaskingVirtualDiskToHosts")
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

// SetSupportsMirrorLocal sets the value of SupportsMirrorLocal for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsMirrorLocal(value bool) (err error) {
	return instance.SetProperty("SupportsMirrorLocal", (value))
}

// GetSupportsMirrorLocal gets the value of SupportsMirrorLocal for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsMirrorLocal() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsMirrorLocal")
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

// SetSupportsMirrorRemote sets the value of SupportsMirrorRemote for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsMirrorRemote(value bool) (err error) {
	return instance.SetProperty("SupportsMirrorRemote", (value))
}

// GetSupportsMirrorRemote gets the value of SupportsMirrorRemote for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsMirrorRemote() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsMirrorRemote")
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

// SetSupportsMultipleResiliencySettingsPerStoragePool sets the value of SupportsMultipleResiliencySettingsPerStoragePool for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsMultipleResiliencySettingsPerStoragePool(value bool) (err error) {
	return instance.SetProperty("SupportsMultipleResiliencySettingsPerStoragePool", (value))
}

// GetSupportsMultipleResiliencySettingsPerStoragePool gets the value of SupportsMultipleResiliencySettingsPerStoragePool for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsMultipleResiliencySettingsPerStoragePool() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsMultipleResiliencySettingsPerStoragePool")
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

// SetSupportsSnapshotLocal sets the value of SupportsSnapshotLocal for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsSnapshotLocal(value bool) (err error) {
	return instance.SetProperty("SupportsSnapshotLocal", (value))
}

// GetSupportsSnapshotLocal gets the value of SupportsSnapshotLocal for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsSnapshotLocal() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsSnapshotLocal")
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

// SetSupportsSnapshotRemote sets the value of SupportsSnapshotRemote for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsSnapshotRemote(value bool) (err error) {
	return instance.SetProperty("SupportsSnapshotRemote", (value))
}

// GetSupportsSnapshotRemote gets the value of SupportsSnapshotRemote for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsSnapshotRemote() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsSnapshotRemote")
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

// SetSupportsStoragePoolAddPhysicalDisk sets the value of SupportsStoragePoolAddPhysicalDisk for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStoragePoolAddPhysicalDisk(value bool) (err error) {
	return instance.SetProperty("SupportsStoragePoolAddPhysicalDisk", (value))
}

// GetSupportsStoragePoolAddPhysicalDisk gets the value of SupportsStoragePoolAddPhysicalDisk for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStoragePoolAddPhysicalDisk() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStoragePoolAddPhysicalDisk")
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

// SetSupportsStoragePoolCreation sets the value of SupportsStoragePoolCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStoragePoolCreation(value bool) (err error) {
	return instance.SetProperty("SupportsStoragePoolCreation", (value))
}

// GetSupportsStoragePoolCreation gets the value of SupportsStoragePoolCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStoragePoolCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStoragePoolCreation")
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

// SetSupportsStoragePoolDeletion sets the value of SupportsStoragePoolDeletion for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStoragePoolDeletion(value bool) (err error) {
	return instance.SetProperty("SupportsStoragePoolDeletion", (value))
}

// GetSupportsStoragePoolDeletion gets the value of SupportsStoragePoolDeletion for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStoragePoolDeletion() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStoragePoolDeletion")
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

// SetSupportsStoragePoolFriendlyNameModification sets the value of SupportsStoragePoolFriendlyNameModification for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStoragePoolFriendlyNameModification(value bool) (err error) {
	return instance.SetProperty("SupportsStoragePoolFriendlyNameModification", (value))
}

// GetSupportsStoragePoolFriendlyNameModification gets the value of SupportsStoragePoolFriendlyNameModification for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStoragePoolFriendlyNameModification() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStoragePoolFriendlyNameModification")
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

// SetSupportsStoragePoolRemovePhysicalDisk sets the value of SupportsStoragePoolRemovePhysicalDisk for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStoragePoolRemovePhysicalDisk(value bool) (err error) {
	return instance.SetProperty("SupportsStoragePoolRemovePhysicalDisk", (value))
}

// GetSupportsStoragePoolRemovePhysicalDisk gets the value of SupportsStoragePoolRemovePhysicalDisk for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStoragePoolRemovePhysicalDisk() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStoragePoolRemovePhysicalDisk")
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

// SetSupportsStorageTierCreation sets the value of SupportsStorageTierCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStorageTierCreation(value bool) (err error) {
	return instance.SetProperty("SupportsStorageTierCreation", (value))
}

// GetSupportsStorageTierCreation gets the value of SupportsStorageTierCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStorageTierCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStorageTierCreation")
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

// SetSupportsStorageTierDeletion sets the value of SupportsStorageTierDeletion for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStorageTierDeletion(value bool) (err error) {
	return instance.SetProperty("SupportsStorageTierDeletion", (value))
}

// GetSupportsStorageTierDeletion gets the value of SupportsStorageTierDeletion for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStorageTierDeletion() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStorageTierDeletion")
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

// SetSupportsStorageTieredVirtualDiskCreation sets the value of SupportsStorageTieredVirtualDiskCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStorageTieredVirtualDiskCreation(value bool) (err error) {
	return instance.SetProperty("SupportsStorageTieredVirtualDiskCreation", (value))
}

// GetSupportsStorageTieredVirtualDiskCreation gets the value of SupportsStorageTieredVirtualDiskCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStorageTieredVirtualDiskCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStorageTieredVirtualDiskCreation")
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

// SetSupportsStorageTierFriendlyNameModification sets the value of SupportsStorageTierFriendlyNameModification for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStorageTierFriendlyNameModification(value bool) (err error) {
	return instance.SetProperty("SupportsStorageTierFriendlyNameModification", (value))
}

// GetSupportsStorageTierFriendlyNameModification gets the value of SupportsStorageTierFriendlyNameModification for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStorageTierFriendlyNameModification() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStorageTierFriendlyNameModification")
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

// SetSupportsStorageTierResize sets the value of SupportsStorageTierResize for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsStorageTierResize(value bool) (err error) {
	return instance.SetProperty("SupportsStorageTierResize", (value))
}

// GetSupportsStorageTierResize gets the value of SupportsStorageTierResize for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsStorageTierResize() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsStorageTierResize")
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

// SetSupportsVirtualDiskCapacityExpansion sets the value of SupportsVirtualDiskCapacityExpansion for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskCapacityExpansion(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskCapacityExpansion", (value))
}

// GetSupportsVirtualDiskCapacityExpansion gets the value of SupportsVirtualDiskCapacityExpansion for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskCapacityExpansion() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskCapacityExpansion")
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

// SetSupportsVirtualDiskCapacityReduction sets the value of SupportsVirtualDiskCapacityReduction for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskCapacityReduction(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskCapacityReduction", (value))
}

// GetSupportsVirtualDiskCapacityReduction gets the value of SupportsVirtualDiskCapacityReduction for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskCapacityReduction() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskCapacityReduction")
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

// SetSupportsVirtualDiskCreation sets the value of SupportsVirtualDiskCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskCreation(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskCreation", (value))
}

// GetSupportsVirtualDiskCreation gets the value of SupportsVirtualDiskCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskCreation")
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

// SetSupportsVirtualDiskDeletion sets the value of SupportsVirtualDiskDeletion for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskDeletion(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskDeletion", (value))
}

// GetSupportsVirtualDiskDeletion gets the value of SupportsVirtualDiskDeletion for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskDeletion() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskDeletion")
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

// SetSupportsVirtualDiskModification sets the value of SupportsVirtualDiskModification for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskModification(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskModification", (value))
}

// GetSupportsVirtualDiskModification gets the value of SupportsVirtualDiskModification for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskModification() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskModification")
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

// SetSupportsVirtualDiskRepair sets the value of SupportsVirtualDiskRepair for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVirtualDiskRepair(value bool) (err error) {
	return instance.SetProperty("SupportsVirtualDiskRepair", (value))
}

// GetSupportsVirtualDiskRepair gets the value of SupportsVirtualDiskRepair for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVirtualDiskRepair() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVirtualDiskRepair")
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

// SetSupportsVolumeCreation sets the value of SupportsVolumeCreation for the instance
func (instance *MSFT_StorageSubSystem) SetPropertySupportsVolumeCreation(value bool) (err error) {
	return instance.SetProperty("SupportsVolumeCreation", (value))
}

// GetSupportsVolumeCreation gets the value of SupportsVolumeCreation for the instance
func (instance *MSFT_StorageSubSystem) GetPropertySupportsVolumeCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsVolumeCreation")
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

// SetTag sets the value of Tag for the instance
func (instance *MSFT_StorageSubSystem) SetPropertyTag(value string) (err error) {
	return instance.SetProperty("Tag", (value))
}

// GetTag gets the value of Tag for the instance
func (instance *MSFT_StorageSubSystem) GetPropertyTag() (value string, err error) {
	retValue, err := instance.GetProperty("Tag")
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

// <param name="AutoWriteCacheSize" type="bool "></param>
// <param name="EnclosureAwareDefault" type="bool "></param>
// <param name="FaultDomainAwarenessDefault" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="LogicalSectorSizeDefault" type="uint64 "></param>
// <param name="MediaTypeDefault" type="uint16 "></param>
// <param name="OtherUsageDescription" type="string "></param>
// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="ProvisioningTypeDefault" type="uint16 "></param>
// <param name="ResiliencySettingNameDefault" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Usage" type="uint16 "></param>
// <param name="Version" type="uint16 "></param>
// <param name="WriteCacheSizeDefault" type="uint64 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedStoragePool" type="MSFT_StoragePool "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) CreateStoragePool( /* IN */ FriendlyName string,
	/* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ ResiliencySettingNameDefault string,
	/* IN */ ProvisioningTypeDefault uint16,
	/* IN */ MediaTypeDefault uint16,
	/* IN */ LogicalSectorSizeDefault uint64,
	/* IN */ EnclosureAwareDefault bool,
	/* IN */ FaultDomainAwarenessDefault uint16,
	/* IN */ WriteCacheSizeDefault uint64,
	/* IN */ AutoWriteCacheSize bool,
	/* IN */ Version uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStoragePool MSFT_StoragePool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateStoragePool", FriendlyName, Usage, OtherUsageDescription, PhysicalDisks, ResiliencySettingNameDefault, ProvisioningTypeDefault, MediaTypeDefault, LogicalSectorSizeDefault, EnclosureAwareDefault, FaultDomainAwarenessDefault, WriteCacheSizeDefault, AutoWriteCacheSize, Version, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="IsEnclosureAware" type="bool "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="OtherUsageDescription" type="string "></param>
// <param name="ParityLayout" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="RequestNoSinglePointOfFailure" type="bool "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>
// <param name="Usage" type="uint16 "></param>
// <param name="UseMaximumSize" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="Size" type="uint64 "></param>
func (instance *MSFT_StorageSubSystem) CreateVirtualDisk( /* IN */ FriendlyName string,
	/* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* IN/OUT */ Size uint64,
	/* IN */ UseMaximumSize bool,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ Interleave uint64,
	/* IN */ ParityLayout uint16,
	/* IN */ RequestNoSinglePointOfFailure bool,
	/* IN */ IsEnclosureAware bool,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ProvisioningType uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedVirtualDisk MSFT_VirtualDisk,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateVirtualDisk", FriendlyName, Usage, OtherUsageDescription, UseMaximumSize, NumberOfDataCopies, PhysicalDiskRedundancy, NumberOfColumns, Interleave, ParityLayout, RequestNoSinglePointOfFailure, IsEnclosureAware, FaultDomainAwareness, ProvisioningType, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DeviceAccesses" type="uint16 []"></param>
// <param name="DeviceNumbers" type="string []"></param>
// <param name="FriendlyName" type="string "></param>
// <param name="HostType" type="uint16 "></param>
// <param name="InitiatorAddresses" type="string []"></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="TargetPortAddresses" type="string []"></param>
// <param name="VirtualDiskNames" type="string []"></param>

// <param name="CreatedMaskingSet" type="MSFT_MaskingSet "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) CreateMaskingSet( /* IN */ FriendlyName string,
	/* IN */ VirtualDiskNames []string,
	/* IN */ DeviceAccesses []uint16,
	/* IN */ DeviceNumbers []string,
	/* IN */ TargetPortAddresses []string,
	/* IN */ InitiatorAddresses []string,
	/* IN */ HostType uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ CreatedMaskingSet MSFT_MaskingSet,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateMaskingSet", FriendlyName, VirtualDiskNames, DeviceAccesses, DeviceNumbers, TargetPortAddresses, InitiatorAddresses, HostType, RunAsJob)
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
func (instance *MSFT_StorageSubSystem) GetSecurityDescriptor( /* OUT */ SecurityDescriptor string,
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
func (instance *MSFT_StorageSubSystem) SetSecurityDescriptor( /* IN */ SecurityDescriptor string,
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

// <param name="Description" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) SetDescription( /* IN */ Description string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetDescription", Description)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AutomaticClusteringEnabled" type="bool "></param>
// <param name="FaultDomainAwarenessDefault" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) SetAttributes( /* IN */ AutomaticClusteringEnabled bool,
	/* IN */ FaultDomainAwarenessDefault uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", AutomaticClusteringEnabled, FaultDomainAwarenessDefault)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>
// <param name="RecoveryPointObjective" type="uint32 "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="SourceGroup" type="MSFT_ReplicationGroup "></param>
// <param name="SourceGroupSettings" type="MSFT_ReplicationSettings "></param>
// <param name="SourceReplicationGroupDescription" type="string "></param>
// <param name="SourceReplicationGroupFriendlyName" type="string "></param>
// <param name="SourceStorageElements" type="MSFT_StorageObject []"></param>
// <param name="SyncType" type="uint16 "></param>
// <param name="TargetGroup" type="MSFT_ReplicationGroup "></param>
// <param name="TargetGroupSettings" type="MSFT_ReplicationSettings "></param>
// <param name="TargetReplicationGroupDescription" type="string "></param>
// <param name="TargetReplicationGroupFriendlyName" type="string "></param>
// <param name="TargetStorageElements" type="MSFT_StorageObject []"></param>
// <param name="TargetStoragePool" type="MSFT_StoragePool "></param>
// <param name="TargetStoragePools" type="MSFT_StoragePool []"></param>
// <param name="TargetStorageSubsystem" type="MSFT_ReplicaPeer "></param>

// <param name="CreatedReplicaPeer" type="MSFT_ReplicaPeer "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SourceGroup" type="MSFT_ReplicationGroup "></param>
func (instance *MSFT_StorageSubSystem) CreateReplicationRelationship( /* IN */ FriendlyName string,
	/* IN */ SyncType uint16,
	/* IN */ TargetStorageSubsystem MSFT_ReplicaPeer,
	/* IN */ SourceReplicationGroupFriendlyName string,
	/* IN */ SourceReplicationGroupDescription string,
	/* IN */ SourceStorageElements []MSFT_StorageObject,
	/* IN */ SourceGroupSettings MSFT_ReplicationSettings,
	/* IN */ TargetReplicationGroupFriendlyName string,
	/* IN */ TargetReplicationGroupDescription string,
	/* IN */ TargetStorageElements []MSFT_StorageObject,
	/* IN */ TargetStoragePool MSFT_StoragePool,
	/* IN */ TargetStoragePools []MSFT_StoragePool,
	/* IN */ TargetGroupSettings MSFT_ReplicationSettings,
	/* IN */ RecoveryPointObjective uint32,
	/* IN */ RunAsJob bool,
	/* IN/OUT */ SourceGroup MSFT_ReplicationGroup,
	/* IN */ TargetGroup MSFT_ReplicationGroup,
	/* OUT */ CreatedReplicaPeer MSFT_ReplicaPeer,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateReplicationRelationship", FriendlyName, SyncType, TargetStorageSubsystem, SourceReplicationGroupFriendlyName, SourceReplicationGroupDescription, SourceStorageElements, SourceGroupSettings, TargetReplicationGroupFriendlyName, TargetReplicationGroupDescription, TargetStorageElements, TargetStoragePool, TargetStoragePools, TargetGroupSettings, RecoveryPointObjective, RunAsJob, TargetGroup)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>
// <param name="SourceReplicationGroup" type="MSFT_ReplicationGroup "></param>
// <param name="TargetGroupReplicaPeer" type="MSFT_ReplicaPeer "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) DeleteReplicationRelationship( /* IN */ SourceReplicationGroup MSFT_ReplicationGroup,
	/* IN */ TargetGroupReplicaPeer MSFT_ReplicaPeer,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteReplicationRelationship", SourceReplicationGroup, TargetGroupReplicaPeer, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Description" type="string "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="ReplicationSettings" type="MSFT_ReplicationSettings "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="StorageElements" type="MSFT_StorageObject []"></param>

// <param name="CreatedReplicationGroup" type="MSFT_ReplicationGroup "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) CreateReplicationGroup( /* IN */ FriendlyName string,
	/* IN */ Description string,
	/* IN */ StorageElements []MSFT_StorageObject,
	/* IN */ ReplicationSettings MSFT_ReplicationSettings,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ CreatedReplicationGroup MSFT_ReplicationGroup,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateReplicationGroup", FriendlyName, Description, StorageElements, ReplicationSettings, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FileSharingProtocols" type="uint16 []"></param>
// <param name="FriendlyName" type="string "></param>
// <param name="HostNames" type="string []"></param>
// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedFileServer" type="MSFT_FileServer "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) CreateFileServer( /* IN */ FriendlyName string,
	/* IN */ FileSharingProtocols []uint16,
	/* IN */ HostNames []string,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedFileServer MSFT_FileServer,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateFileServer", FriendlyName, FileSharingProtocols, HostNames, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ActivityId" type="string "></param>
// <param name="CopyExistingInfoOnly" type="bool "></param>
// <param name="DestinationPath" type="string "></param>
// <param name="ExcludeDiagnosticLog" type="bool "></param>
// <param name="ExcludeOperationalLog" type="bool "></param>
// <param name="IncludeLiveDump" type="bool "></param>
// <param name="TimeSpan" type="uint32 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) GetDiagnosticInfo( /* IN */ DestinationPath string,
	/* IN */ TimeSpan uint32,
	/* IN */ ActivityId string,
	/* IN */ ExcludeOperationalLog bool,
	/* IN */ ExcludeDiagnosticLog bool,
	/* IN */ IncludeLiveDump bool,
	/* IN */ CopyExistingInfoOnly bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetDiagnosticInfo", DestinationPath, TimeSpan, ActivityId, ExcludeOperationalLog, ExcludeDiagnosticLog, IncludeLiveDump, CopyExistingInfoOnly)
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
func (instance *MSFT_StorageSubSystem) ClearDiagnosticInfo( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("ClearDiagnosticInfo")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Level" type="uint16 "></param>
// <param name="MaxLogSize" type="uint64 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) StartDiagnosticLog( /* IN */ Level uint16,
	/* IN */ MaxLogSize uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("StartDiagnosticLog", Level, MaxLogSize)
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
func (instance *MSFT_StorageSubSystem) StopDiagnosticLog( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("StopDiagnosticLog")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DiagnoseResults" type="MSFT_StorageDiagnoseResult []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) Diagnose( /* OUT */ DiagnoseResults []MSFT_StorageDiagnoseResult,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Diagnose")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ActionResults" type="MSFT_HealthAction []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSubSystem) GetActions( /* OUT */ ActionResults []MSFT_HealthAction,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetActions")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
