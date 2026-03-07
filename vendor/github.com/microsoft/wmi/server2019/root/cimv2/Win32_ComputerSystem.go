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

// Win32_ComputerSystem struct
type Win32_ComputerSystem struct {
	*CIM_UnitaryComputerSystem

	//
	AdminPasswordStatus uint16

	//
	AutomaticManagedPagefile bool

	//
	AutomaticResetBootOption bool

	//
	AutomaticResetCapability bool

	//
	BootOptionOnLimit uint16

	//
	BootOptionOnWatchDog uint16

	//
	BootROMSupported bool

	//
	BootStatus []uint16

	//
	BootupState string

	//
	ChassisBootupState uint16

	//
	ChassisSKUNumber string

	//
	CurrentTimeZone int16

	//
	DaylightInEffect bool

	//
	DNSHostName string

	//
	Domain string

	//
	DomainRole uint16

	//
	EnableDaylightSavingsTime bool

	//
	FrontPanelResetStatus uint16

	//
	HypervisorPresent bool

	//
	InfraredSupported bool

	//
	KeyboardPasswordStatus uint16

	//
	Manufacturer string

	//
	Model string

	//
	NetworkServerModeEnabled bool

	//
	NumberOfLogicalProcessors uint32

	//
	NumberOfProcessors uint32

	//
	OEMLogoBitmap []uint8

	//
	OEMStringArray []string

	//
	PartOfDomain bool

	//
	PauseAfterReset int64

	//
	PCSystemType uint16

	//
	PCSystemTypeEx uint16

	//
	PowerOnPasswordStatus uint16

	//
	PowerSupplyState uint16

	//
	ResetCount int16

	//
	ResetLimit int16

	//
	SupportContactDescription []string

	//
	SystemFamily string

	//
	SystemSKUNumber string

	//
	SystemStartupDelay uint16

	//
	SystemStartupOptions []string

	//
	SystemStartupSetting uint8

	//
	SystemType string

	//
	ThermalState uint16

	//
	TotalPhysicalMemory uint64

	//
	UserName string

	//
	WakeUpType uint16

	//
	Workgroup string
}

func NewWin32_ComputerSystemEx1(instance *cim.WmiInstance) (newInstance *Win32_ComputerSystem, err error) {
	tmp, err := NewCIM_UnitaryComputerSystemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystem{
		CIM_UnitaryComputerSystem: tmp,
	}
	return
}

func NewWin32_ComputerSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComputerSystem, err error) {
	tmp, err := NewCIM_UnitaryComputerSystemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystem{
		CIM_UnitaryComputerSystem: tmp,
	}
	return
}

// SetAdminPasswordStatus sets the value of AdminPasswordStatus for the instance
func (instance *Win32_ComputerSystem) SetPropertyAdminPasswordStatus(value uint16) (err error) {
	return instance.SetProperty("AdminPasswordStatus", (value))
}

// GetAdminPasswordStatus gets the value of AdminPasswordStatus for the instance
func (instance *Win32_ComputerSystem) GetPropertyAdminPasswordStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("AdminPasswordStatus")
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

// SetAutomaticManagedPagefile sets the value of AutomaticManagedPagefile for the instance
func (instance *Win32_ComputerSystem) SetPropertyAutomaticManagedPagefile(value bool) (err error) {
	return instance.SetProperty("AutomaticManagedPagefile", (value))
}

// GetAutomaticManagedPagefile gets the value of AutomaticManagedPagefile for the instance
func (instance *Win32_ComputerSystem) GetPropertyAutomaticManagedPagefile() (value bool, err error) {
	retValue, err := instance.GetProperty("AutomaticManagedPagefile")
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

// SetAutomaticResetBootOption sets the value of AutomaticResetBootOption for the instance
func (instance *Win32_ComputerSystem) SetPropertyAutomaticResetBootOption(value bool) (err error) {
	return instance.SetProperty("AutomaticResetBootOption", (value))
}

// GetAutomaticResetBootOption gets the value of AutomaticResetBootOption for the instance
func (instance *Win32_ComputerSystem) GetPropertyAutomaticResetBootOption() (value bool, err error) {
	retValue, err := instance.GetProperty("AutomaticResetBootOption")
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

// SetAutomaticResetCapability sets the value of AutomaticResetCapability for the instance
func (instance *Win32_ComputerSystem) SetPropertyAutomaticResetCapability(value bool) (err error) {
	return instance.SetProperty("AutomaticResetCapability", (value))
}

// GetAutomaticResetCapability gets the value of AutomaticResetCapability for the instance
func (instance *Win32_ComputerSystem) GetPropertyAutomaticResetCapability() (value bool, err error) {
	retValue, err := instance.GetProperty("AutomaticResetCapability")
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

// SetBootOptionOnLimit sets the value of BootOptionOnLimit for the instance
func (instance *Win32_ComputerSystem) SetPropertyBootOptionOnLimit(value uint16) (err error) {
	return instance.SetProperty("BootOptionOnLimit", (value))
}

// GetBootOptionOnLimit gets the value of BootOptionOnLimit for the instance
func (instance *Win32_ComputerSystem) GetPropertyBootOptionOnLimit() (value uint16, err error) {
	retValue, err := instance.GetProperty("BootOptionOnLimit")
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

// SetBootOptionOnWatchDog sets the value of BootOptionOnWatchDog for the instance
func (instance *Win32_ComputerSystem) SetPropertyBootOptionOnWatchDog(value uint16) (err error) {
	return instance.SetProperty("BootOptionOnWatchDog", (value))
}

// GetBootOptionOnWatchDog gets the value of BootOptionOnWatchDog for the instance
func (instance *Win32_ComputerSystem) GetPropertyBootOptionOnWatchDog() (value uint16, err error) {
	retValue, err := instance.GetProperty("BootOptionOnWatchDog")
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

// SetBootROMSupported sets the value of BootROMSupported for the instance
func (instance *Win32_ComputerSystem) SetPropertyBootROMSupported(value bool) (err error) {
	return instance.SetProperty("BootROMSupported", (value))
}

// GetBootROMSupported gets the value of BootROMSupported for the instance
func (instance *Win32_ComputerSystem) GetPropertyBootROMSupported() (value bool, err error) {
	retValue, err := instance.GetProperty("BootROMSupported")
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

// SetBootStatus sets the value of BootStatus for the instance
func (instance *Win32_ComputerSystem) SetPropertyBootStatus(value []uint16) (err error) {
	return instance.SetProperty("BootStatus", (value))
}

// GetBootStatus gets the value of BootStatus for the instance
func (instance *Win32_ComputerSystem) GetPropertyBootStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("BootStatus")
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

// SetBootupState sets the value of BootupState for the instance
func (instance *Win32_ComputerSystem) SetPropertyBootupState(value string) (err error) {
	return instance.SetProperty("BootupState", (value))
}

// GetBootupState gets the value of BootupState for the instance
func (instance *Win32_ComputerSystem) GetPropertyBootupState() (value string, err error) {
	retValue, err := instance.GetProperty("BootupState")
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

// SetChassisBootupState sets the value of ChassisBootupState for the instance
func (instance *Win32_ComputerSystem) SetPropertyChassisBootupState(value uint16) (err error) {
	return instance.SetProperty("ChassisBootupState", (value))
}

// GetChassisBootupState gets the value of ChassisBootupState for the instance
func (instance *Win32_ComputerSystem) GetPropertyChassisBootupState() (value uint16, err error) {
	retValue, err := instance.GetProperty("ChassisBootupState")
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

// SetChassisSKUNumber sets the value of ChassisSKUNumber for the instance
func (instance *Win32_ComputerSystem) SetPropertyChassisSKUNumber(value string) (err error) {
	return instance.SetProperty("ChassisSKUNumber", (value))
}

// GetChassisSKUNumber gets the value of ChassisSKUNumber for the instance
func (instance *Win32_ComputerSystem) GetPropertyChassisSKUNumber() (value string, err error) {
	retValue, err := instance.GetProperty("ChassisSKUNumber")
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

// SetCurrentTimeZone sets the value of CurrentTimeZone for the instance
func (instance *Win32_ComputerSystem) SetPropertyCurrentTimeZone(value int16) (err error) {
	return instance.SetProperty("CurrentTimeZone", (value))
}

// GetCurrentTimeZone gets the value of CurrentTimeZone for the instance
func (instance *Win32_ComputerSystem) GetPropertyCurrentTimeZone() (value int16, err error) {
	retValue, err := instance.GetProperty("CurrentTimeZone")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetDaylightInEffect sets the value of DaylightInEffect for the instance
func (instance *Win32_ComputerSystem) SetPropertyDaylightInEffect(value bool) (err error) {
	return instance.SetProperty("DaylightInEffect", (value))
}

// GetDaylightInEffect gets the value of DaylightInEffect for the instance
func (instance *Win32_ComputerSystem) GetPropertyDaylightInEffect() (value bool, err error) {
	retValue, err := instance.GetProperty("DaylightInEffect")
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

// SetDNSHostName sets the value of DNSHostName for the instance
func (instance *Win32_ComputerSystem) SetPropertyDNSHostName(value string) (err error) {
	return instance.SetProperty("DNSHostName", (value))
}

// GetDNSHostName gets the value of DNSHostName for the instance
func (instance *Win32_ComputerSystem) GetPropertyDNSHostName() (value string, err error) {
	retValue, err := instance.GetProperty("DNSHostName")
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

// SetDomain sets the value of Domain for the instance
func (instance *Win32_ComputerSystem) SetPropertyDomain(value string) (err error) {
	return instance.SetProperty("Domain", (value))
}

// GetDomain gets the value of Domain for the instance
func (instance *Win32_ComputerSystem) GetPropertyDomain() (value string, err error) {
	retValue, err := instance.GetProperty("Domain")
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

// SetDomainRole sets the value of DomainRole for the instance
func (instance *Win32_ComputerSystem) SetPropertyDomainRole(value uint16) (err error) {
	return instance.SetProperty("DomainRole", (value))
}

// GetDomainRole gets the value of DomainRole for the instance
func (instance *Win32_ComputerSystem) GetPropertyDomainRole() (value uint16, err error) {
	retValue, err := instance.GetProperty("DomainRole")
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

// SetEnableDaylightSavingsTime sets the value of EnableDaylightSavingsTime for the instance
func (instance *Win32_ComputerSystem) SetPropertyEnableDaylightSavingsTime(value bool) (err error) {
	return instance.SetProperty("EnableDaylightSavingsTime", (value))
}

// GetEnableDaylightSavingsTime gets the value of EnableDaylightSavingsTime for the instance
func (instance *Win32_ComputerSystem) GetPropertyEnableDaylightSavingsTime() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableDaylightSavingsTime")
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

// SetFrontPanelResetStatus sets the value of FrontPanelResetStatus for the instance
func (instance *Win32_ComputerSystem) SetPropertyFrontPanelResetStatus(value uint16) (err error) {
	return instance.SetProperty("FrontPanelResetStatus", (value))
}

// GetFrontPanelResetStatus gets the value of FrontPanelResetStatus for the instance
func (instance *Win32_ComputerSystem) GetPropertyFrontPanelResetStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("FrontPanelResetStatus")
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

// SetHypervisorPresent sets the value of HypervisorPresent for the instance
func (instance *Win32_ComputerSystem) SetPropertyHypervisorPresent(value bool) (err error) {
	return instance.SetProperty("HypervisorPresent", (value))
}

// GetHypervisorPresent gets the value of HypervisorPresent for the instance
func (instance *Win32_ComputerSystem) GetPropertyHypervisorPresent() (value bool, err error) {
	retValue, err := instance.GetProperty("HypervisorPresent")
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

// SetInfraredSupported sets the value of InfraredSupported for the instance
func (instance *Win32_ComputerSystem) SetPropertyInfraredSupported(value bool) (err error) {
	return instance.SetProperty("InfraredSupported", (value))
}

// GetInfraredSupported gets the value of InfraredSupported for the instance
func (instance *Win32_ComputerSystem) GetPropertyInfraredSupported() (value bool, err error) {
	retValue, err := instance.GetProperty("InfraredSupported")
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

// SetKeyboardPasswordStatus sets the value of KeyboardPasswordStatus for the instance
func (instance *Win32_ComputerSystem) SetPropertyKeyboardPasswordStatus(value uint16) (err error) {
	return instance.SetProperty("KeyboardPasswordStatus", (value))
}

// GetKeyboardPasswordStatus gets the value of KeyboardPasswordStatus for the instance
func (instance *Win32_ComputerSystem) GetPropertyKeyboardPasswordStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("KeyboardPasswordStatus")
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
func (instance *Win32_ComputerSystem) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_ComputerSystem) GetPropertyManufacturer() (value string, err error) {
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

// SetModel sets the value of Model for the instance
func (instance *Win32_ComputerSystem) SetPropertyModel(value string) (err error) {
	return instance.SetProperty("Model", (value))
}

// GetModel gets the value of Model for the instance
func (instance *Win32_ComputerSystem) GetPropertyModel() (value string, err error) {
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

// SetNetworkServerModeEnabled sets the value of NetworkServerModeEnabled for the instance
func (instance *Win32_ComputerSystem) SetPropertyNetworkServerModeEnabled(value bool) (err error) {
	return instance.SetProperty("NetworkServerModeEnabled", (value))
}

// GetNetworkServerModeEnabled gets the value of NetworkServerModeEnabled for the instance
func (instance *Win32_ComputerSystem) GetPropertyNetworkServerModeEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("NetworkServerModeEnabled")
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

// SetNumberOfLogicalProcessors sets the value of NumberOfLogicalProcessors for the instance
func (instance *Win32_ComputerSystem) SetPropertyNumberOfLogicalProcessors(value uint32) (err error) {
	return instance.SetProperty("NumberOfLogicalProcessors", (value))
}

// GetNumberOfLogicalProcessors gets the value of NumberOfLogicalProcessors for the instance
func (instance *Win32_ComputerSystem) GetPropertyNumberOfLogicalProcessors() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfLogicalProcessors")
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

// SetNumberOfProcessors sets the value of NumberOfProcessors for the instance
func (instance *Win32_ComputerSystem) SetPropertyNumberOfProcessors(value uint32) (err error) {
	return instance.SetProperty("NumberOfProcessors", (value))
}

// GetNumberOfProcessors gets the value of NumberOfProcessors for the instance
func (instance *Win32_ComputerSystem) GetPropertyNumberOfProcessors() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfProcessors")
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

// SetOEMLogoBitmap sets the value of OEMLogoBitmap for the instance
func (instance *Win32_ComputerSystem) SetPropertyOEMLogoBitmap(value []uint8) (err error) {
	return instance.SetProperty("OEMLogoBitmap", (value))
}

// GetOEMLogoBitmap gets the value of OEMLogoBitmap for the instance
func (instance *Win32_ComputerSystem) GetPropertyOEMLogoBitmap() (value []uint8, err error) {
	retValue, err := instance.GetProperty("OEMLogoBitmap")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetOEMStringArray sets the value of OEMStringArray for the instance
func (instance *Win32_ComputerSystem) SetPropertyOEMStringArray(value []string) (err error) {
	return instance.SetProperty("OEMStringArray", (value))
}

// GetOEMStringArray gets the value of OEMStringArray for the instance
func (instance *Win32_ComputerSystem) GetPropertyOEMStringArray() (value []string, err error) {
	retValue, err := instance.GetProperty("OEMStringArray")
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

// SetPartOfDomain sets the value of PartOfDomain for the instance
func (instance *Win32_ComputerSystem) SetPropertyPartOfDomain(value bool) (err error) {
	return instance.SetProperty("PartOfDomain", (value))
}

// GetPartOfDomain gets the value of PartOfDomain for the instance
func (instance *Win32_ComputerSystem) GetPropertyPartOfDomain() (value bool, err error) {
	retValue, err := instance.GetProperty("PartOfDomain")
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

// SetPauseAfterReset sets the value of PauseAfterReset for the instance
func (instance *Win32_ComputerSystem) SetPropertyPauseAfterReset(value int64) (err error) {
	return instance.SetProperty("PauseAfterReset", (value))
}

// GetPauseAfterReset gets the value of PauseAfterReset for the instance
func (instance *Win32_ComputerSystem) GetPropertyPauseAfterReset() (value int64, err error) {
	retValue, err := instance.GetProperty("PauseAfterReset")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetPCSystemType sets the value of PCSystemType for the instance
func (instance *Win32_ComputerSystem) SetPropertyPCSystemType(value uint16) (err error) {
	return instance.SetProperty("PCSystemType", (value))
}

// GetPCSystemType gets the value of PCSystemType for the instance
func (instance *Win32_ComputerSystem) GetPropertyPCSystemType() (value uint16, err error) {
	retValue, err := instance.GetProperty("PCSystemType")
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

// SetPCSystemTypeEx sets the value of PCSystemTypeEx for the instance
func (instance *Win32_ComputerSystem) SetPropertyPCSystemTypeEx(value uint16) (err error) {
	return instance.SetProperty("PCSystemTypeEx", (value))
}

// GetPCSystemTypeEx gets the value of PCSystemTypeEx for the instance
func (instance *Win32_ComputerSystem) GetPropertyPCSystemTypeEx() (value uint16, err error) {
	retValue, err := instance.GetProperty("PCSystemTypeEx")
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

// SetPowerOnPasswordStatus sets the value of PowerOnPasswordStatus for the instance
func (instance *Win32_ComputerSystem) SetPropertyPowerOnPasswordStatus(value uint16) (err error) {
	return instance.SetProperty("PowerOnPasswordStatus", (value))
}

// GetPowerOnPasswordStatus gets the value of PowerOnPasswordStatus for the instance
func (instance *Win32_ComputerSystem) GetPropertyPowerOnPasswordStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("PowerOnPasswordStatus")
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

// SetPowerSupplyState sets the value of PowerSupplyState for the instance
func (instance *Win32_ComputerSystem) SetPropertyPowerSupplyState(value uint16) (err error) {
	return instance.SetProperty("PowerSupplyState", (value))
}

// GetPowerSupplyState gets the value of PowerSupplyState for the instance
func (instance *Win32_ComputerSystem) GetPropertyPowerSupplyState() (value uint16, err error) {
	retValue, err := instance.GetProperty("PowerSupplyState")
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

// SetResetCount sets the value of ResetCount for the instance
func (instance *Win32_ComputerSystem) SetPropertyResetCount(value int16) (err error) {
	return instance.SetProperty("ResetCount", (value))
}

// GetResetCount gets the value of ResetCount for the instance
func (instance *Win32_ComputerSystem) GetPropertyResetCount() (value int16, err error) {
	retValue, err := instance.GetProperty("ResetCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetResetLimit sets the value of ResetLimit for the instance
func (instance *Win32_ComputerSystem) SetPropertyResetLimit(value int16) (err error) {
	return instance.SetProperty("ResetLimit", (value))
}

// GetResetLimit gets the value of ResetLimit for the instance
func (instance *Win32_ComputerSystem) GetPropertyResetLimit() (value int16, err error) {
	retValue, err := instance.GetProperty("ResetLimit")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetSupportContactDescription sets the value of SupportContactDescription for the instance
func (instance *Win32_ComputerSystem) SetPropertySupportContactDescription(value []string) (err error) {
	return instance.SetProperty("SupportContactDescription", (value))
}

// GetSupportContactDescription gets the value of SupportContactDescription for the instance
func (instance *Win32_ComputerSystem) GetPropertySupportContactDescription() (value []string, err error) {
	retValue, err := instance.GetProperty("SupportContactDescription")
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

// SetSystemFamily sets the value of SystemFamily for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemFamily(value string) (err error) {
	return instance.SetProperty("SystemFamily", (value))
}

// GetSystemFamily gets the value of SystemFamily for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemFamily() (value string, err error) {
	retValue, err := instance.GetProperty("SystemFamily")
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

// SetSystemSKUNumber sets the value of SystemSKUNumber for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemSKUNumber(value string) (err error) {
	return instance.SetProperty("SystemSKUNumber", (value))
}

// GetSystemSKUNumber gets the value of SystemSKUNumber for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemSKUNumber() (value string, err error) {
	retValue, err := instance.GetProperty("SystemSKUNumber")
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

// SetSystemStartupDelay sets the value of SystemStartupDelay for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemStartupDelay(value uint16) (err error) {
	return instance.SetProperty("SystemStartupDelay", (value))
}

// GetSystemStartupDelay gets the value of SystemStartupDelay for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemStartupDelay() (value uint16, err error) {
	retValue, err := instance.GetProperty("SystemStartupDelay")
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

// SetSystemStartupOptions sets the value of SystemStartupOptions for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemStartupOptions(value []string) (err error) {
	return instance.SetProperty("SystemStartupOptions", (value))
}

// GetSystemStartupOptions gets the value of SystemStartupOptions for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemStartupOptions() (value []string, err error) {
	retValue, err := instance.GetProperty("SystemStartupOptions")
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

// SetSystemStartupSetting sets the value of SystemStartupSetting for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemStartupSetting(value uint8) (err error) {
	return instance.SetProperty("SystemStartupSetting", (value))
}

// GetSystemStartupSetting gets the value of SystemStartupSetting for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemStartupSetting() (value uint8, err error) {
	retValue, err := instance.GetProperty("SystemStartupSetting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetSystemType sets the value of SystemType for the instance
func (instance *Win32_ComputerSystem) SetPropertySystemType(value string) (err error) {
	return instance.SetProperty("SystemType", (value))
}

// GetSystemType gets the value of SystemType for the instance
func (instance *Win32_ComputerSystem) GetPropertySystemType() (value string, err error) {
	retValue, err := instance.GetProperty("SystemType")
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

// SetThermalState sets the value of ThermalState for the instance
func (instance *Win32_ComputerSystem) SetPropertyThermalState(value uint16) (err error) {
	return instance.SetProperty("ThermalState", (value))
}

// GetThermalState gets the value of ThermalState for the instance
func (instance *Win32_ComputerSystem) GetPropertyThermalState() (value uint16, err error) {
	retValue, err := instance.GetProperty("ThermalState")
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

// SetTotalPhysicalMemory sets the value of TotalPhysicalMemory for the instance
func (instance *Win32_ComputerSystem) SetPropertyTotalPhysicalMemory(value uint64) (err error) {
	return instance.SetProperty("TotalPhysicalMemory", (value))
}

// GetTotalPhysicalMemory gets the value of TotalPhysicalMemory for the instance
func (instance *Win32_ComputerSystem) GetPropertyTotalPhysicalMemory() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalPhysicalMemory")
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

// SetUserName sets the value of UserName for the instance
func (instance *Win32_ComputerSystem) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_ComputerSystem) GetPropertyUserName() (value string, err error) {
	retValue, err := instance.GetProperty("UserName")
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

// SetWakeUpType sets the value of WakeUpType for the instance
func (instance *Win32_ComputerSystem) SetPropertyWakeUpType(value uint16) (err error) {
	return instance.SetProperty("WakeUpType", (value))
}

// GetWakeUpType gets the value of WakeUpType for the instance
func (instance *Win32_ComputerSystem) GetPropertyWakeUpType() (value uint16, err error) {
	retValue, err := instance.GetProperty("WakeUpType")
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

// SetWorkgroup sets the value of Workgroup for the instance
func (instance *Win32_ComputerSystem) SetPropertyWorkgroup(value string) (err error) {
	return instance.SetProperty("Workgroup", (value))
}

// GetWorkgroup gets the value of Workgroup for the instance
func (instance *Win32_ComputerSystem) GetPropertyWorkgroup() (value string, err error) {
	retValue, err := instance.GetProperty("Workgroup")
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

// <param name="Name" type="string "></param>
// <param name="Password" type="string "></param>
// <param name="UserName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ComputerSystem) Rename( /* IN */ Name string,
	/* IN */ Password string,
	/* IN */ UserName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Rename", Name, Password, UserName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AccountOU" type="string "></param>
// <param name="FJoinOptions" type="uint32 "></param>
// <param name="Name" type="string "></param>
// <param name="Password" type="string "></param>
// <param name="UserName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ComputerSystem) JoinDomainOrWorkgroup( /* IN */ Name string,
	/* IN */ Password string,
	/* IN */ UserName string,
	/* IN */ AccountOU string,
	/* IN */ FJoinOptions uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("JoinDomainOrWorkgroup", Name, Password, UserName, AccountOU, FJoinOptions)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="FUnjoinOptions" type="uint32 "></param>
// <param name="Password" type="string "></param>
// <param name="UserName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ComputerSystem) UnjoinDomainOrWorkgroup( /* IN */ Password string,
	/* IN */ UserName string,
	/* IN */ FUnjoinOptions uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("UnjoinDomainOrWorkgroup", Password, UserName, FUnjoinOptions)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
