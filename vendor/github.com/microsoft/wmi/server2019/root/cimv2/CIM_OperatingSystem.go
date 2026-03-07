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

// CIM_OperatingSystem struct
type CIM_OperatingSystem struct {
	*CIM_LogicalElement

	//
	CreationClassName string

	//
	CSCreationClassName string

	//
	CSName string

	//
	CurrentTimeZone int16

	//
	Distributed bool

	//
	FreePhysicalMemory uint64

	//
	FreeSpaceInPagingFiles uint64

	//
	FreeVirtualMemory uint64

	//
	LastBootUpTime string

	//
	LocalDateTime string

	//
	MaxNumberOfProcesses uint32

	//
	MaxProcessMemorySize uint64

	//
	NumberOfLicensedUsers uint32

	//
	NumberOfProcesses uint32

	//
	NumberOfUsers uint32

	//
	OSType uint16

	//
	OtherTypeDescription string

	//
	SizeStoredInPagingFiles uint64

	//
	TotalSwapSpaceSize uint64

	//
	TotalVirtualMemorySize uint64

	//
	TotalVisibleMemorySize uint64

	//
	Version string
}

func NewCIM_OperatingSystemEx1(instance *cim.WmiInstance) (newInstance *CIM_OperatingSystem, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_OperatingSystem{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_OperatingSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_OperatingSystem, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_OperatingSystem{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_OperatingSystem) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_OperatingSystem) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_OperatingSystem) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_OperatingSystem) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_OperatingSystem) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_OperatingSystem) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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
func (instance *CIM_OperatingSystem) SetPropertyCurrentTimeZone(value int16) (err error) {
	return instance.SetProperty("CurrentTimeZone", (value))
}

// GetCurrentTimeZone gets the value of CurrentTimeZone for the instance
func (instance *CIM_OperatingSystem) GetPropertyCurrentTimeZone() (value int16, err error) {
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

// SetDistributed sets the value of Distributed for the instance
func (instance *CIM_OperatingSystem) SetPropertyDistributed(value bool) (err error) {
	return instance.SetProperty("Distributed", (value))
}

// GetDistributed gets the value of Distributed for the instance
func (instance *CIM_OperatingSystem) GetPropertyDistributed() (value bool, err error) {
	retValue, err := instance.GetProperty("Distributed")
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

// SetFreePhysicalMemory sets the value of FreePhysicalMemory for the instance
func (instance *CIM_OperatingSystem) SetPropertyFreePhysicalMemory(value uint64) (err error) {
	return instance.SetProperty("FreePhysicalMemory", (value))
}

// GetFreePhysicalMemory gets the value of FreePhysicalMemory for the instance
func (instance *CIM_OperatingSystem) GetPropertyFreePhysicalMemory() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreePhysicalMemory")
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

// SetFreeSpaceInPagingFiles sets the value of FreeSpaceInPagingFiles for the instance
func (instance *CIM_OperatingSystem) SetPropertyFreeSpaceInPagingFiles(value uint64) (err error) {
	return instance.SetProperty("FreeSpaceInPagingFiles", (value))
}

// GetFreeSpaceInPagingFiles gets the value of FreeSpaceInPagingFiles for the instance
func (instance *CIM_OperatingSystem) GetPropertyFreeSpaceInPagingFiles() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreeSpaceInPagingFiles")
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

// SetFreeVirtualMemory sets the value of FreeVirtualMemory for the instance
func (instance *CIM_OperatingSystem) SetPropertyFreeVirtualMemory(value uint64) (err error) {
	return instance.SetProperty("FreeVirtualMemory", (value))
}

// GetFreeVirtualMemory gets the value of FreeVirtualMemory for the instance
func (instance *CIM_OperatingSystem) GetPropertyFreeVirtualMemory() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreeVirtualMemory")
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

// SetLastBootUpTime sets the value of LastBootUpTime for the instance
func (instance *CIM_OperatingSystem) SetPropertyLastBootUpTime(value string) (err error) {
	return instance.SetProperty("LastBootUpTime", (value))
}

// GetLastBootUpTime gets the value of LastBootUpTime for the instance
func (instance *CIM_OperatingSystem) GetPropertyLastBootUpTime() (value string, err error) {
	retValue, err := instance.GetProperty("LastBootUpTime")
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

// SetLocalDateTime sets the value of LocalDateTime for the instance
func (instance *CIM_OperatingSystem) SetPropertyLocalDateTime(value string) (err error) {
	return instance.SetProperty("LocalDateTime", (value))
}

// GetLocalDateTime gets the value of LocalDateTime for the instance
func (instance *CIM_OperatingSystem) GetPropertyLocalDateTime() (value string, err error) {
	retValue, err := instance.GetProperty("LocalDateTime")
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

// SetMaxNumberOfProcesses sets the value of MaxNumberOfProcesses for the instance
func (instance *CIM_OperatingSystem) SetPropertyMaxNumberOfProcesses(value uint32) (err error) {
	return instance.SetProperty("MaxNumberOfProcesses", (value))
}

// GetMaxNumberOfProcesses gets the value of MaxNumberOfProcesses for the instance
func (instance *CIM_OperatingSystem) GetPropertyMaxNumberOfProcesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxNumberOfProcesses")
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

// SetMaxProcessMemorySize sets the value of MaxProcessMemorySize for the instance
func (instance *CIM_OperatingSystem) SetPropertyMaxProcessMemorySize(value uint64) (err error) {
	return instance.SetProperty("MaxProcessMemorySize", (value))
}

// GetMaxProcessMemorySize gets the value of MaxProcessMemorySize for the instance
func (instance *CIM_OperatingSystem) GetPropertyMaxProcessMemorySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxProcessMemorySize")
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

// SetNumberOfLicensedUsers sets the value of NumberOfLicensedUsers for the instance
func (instance *CIM_OperatingSystem) SetPropertyNumberOfLicensedUsers(value uint32) (err error) {
	return instance.SetProperty("NumberOfLicensedUsers", (value))
}

// GetNumberOfLicensedUsers gets the value of NumberOfLicensedUsers for the instance
func (instance *CIM_OperatingSystem) GetPropertyNumberOfLicensedUsers() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfLicensedUsers")
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

// SetNumberOfProcesses sets the value of NumberOfProcesses for the instance
func (instance *CIM_OperatingSystem) SetPropertyNumberOfProcesses(value uint32) (err error) {
	return instance.SetProperty("NumberOfProcesses", (value))
}

// GetNumberOfProcesses gets the value of NumberOfProcesses for the instance
func (instance *CIM_OperatingSystem) GetPropertyNumberOfProcesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfProcesses")
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

// SetNumberOfUsers sets the value of NumberOfUsers for the instance
func (instance *CIM_OperatingSystem) SetPropertyNumberOfUsers(value uint32) (err error) {
	return instance.SetProperty("NumberOfUsers", (value))
}

// GetNumberOfUsers gets the value of NumberOfUsers for the instance
func (instance *CIM_OperatingSystem) GetPropertyNumberOfUsers() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfUsers")
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

// SetOSType sets the value of OSType for the instance
func (instance *CIM_OperatingSystem) SetPropertyOSType(value uint16) (err error) {
	return instance.SetProperty("OSType", (value))
}

// GetOSType gets the value of OSType for the instance
func (instance *CIM_OperatingSystem) GetPropertyOSType() (value uint16, err error) {
	retValue, err := instance.GetProperty("OSType")
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

// SetOtherTypeDescription sets the value of OtherTypeDescription for the instance
func (instance *CIM_OperatingSystem) SetPropertyOtherTypeDescription(value string) (err error) {
	return instance.SetProperty("OtherTypeDescription", (value))
}

// GetOtherTypeDescription gets the value of OtherTypeDescription for the instance
func (instance *CIM_OperatingSystem) GetPropertyOtherTypeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherTypeDescription")
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

// SetSizeStoredInPagingFiles sets the value of SizeStoredInPagingFiles for the instance
func (instance *CIM_OperatingSystem) SetPropertySizeStoredInPagingFiles(value uint64) (err error) {
	return instance.SetProperty("SizeStoredInPagingFiles", (value))
}

// GetSizeStoredInPagingFiles gets the value of SizeStoredInPagingFiles for the instance
func (instance *CIM_OperatingSystem) GetPropertySizeStoredInPagingFiles() (value uint64, err error) {
	retValue, err := instance.GetProperty("SizeStoredInPagingFiles")
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

// SetTotalSwapSpaceSize sets the value of TotalSwapSpaceSize for the instance
func (instance *CIM_OperatingSystem) SetPropertyTotalSwapSpaceSize(value uint64) (err error) {
	return instance.SetProperty("TotalSwapSpaceSize", (value))
}

// GetTotalSwapSpaceSize gets the value of TotalSwapSpaceSize for the instance
func (instance *CIM_OperatingSystem) GetPropertyTotalSwapSpaceSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalSwapSpaceSize")
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

// SetTotalVirtualMemorySize sets the value of TotalVirtualMemorySize for the instance
func (instance *CIM_OperatingSystem) SetPropertyTotalVirtualMemorySize(value uint64) (err error) {
	return instance.SetProperty("TotalVirtualMemorySize", (value))
}

// GetTotalVirtualMemorySize gets the value of TotalVirtualMemorySize for the instance
func (instance *CIM_OperatingSystem) GetPropertyTotalVirtualMemorySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalVirtualMemorySize")
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

// SetTotalVisibleMemorySize sets the value of TotalVisibleMemorySize for the instance
func (instance *CIM_OperatingSystem) SetPropertyTotalVisibleMemorySize(value uint64) (err error) {
	return instance.SetProperty("TotalVisibleMemorySize", (value))
}

// GetTotalVisibleMemorySize gets the value of TotalVisibleMemorySize for the instance
func (instance *CIM_OperatingSystem) GetPropertyTotalVisibleMemorySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalVisibleMemorySize")
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

// SetVersion sets the value of Version for the instance
func (instance *CIM_OperatingSystem) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *CIM_OperatingSystem) GetPropertyVersion() (value string, err error) {
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

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_OperatingSystem) Reboot() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Reboot")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_OperatingSystem) Shutdown() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Shutdown")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
