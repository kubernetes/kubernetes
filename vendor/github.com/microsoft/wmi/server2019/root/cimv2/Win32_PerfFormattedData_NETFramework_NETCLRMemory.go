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

// Win32_PerfFormattedData_NETFramework_NETCLRMemory struct
type Win32_PerfFormattedData_NETFramework_NETCLRMemory struct {
	*Win32_PerfFormattedData

	//
	AllocatedBytesPersec uint32

	//
	FinalizationSurvivors uint32

	//
	Gen0heapsize uint32

	//
	Gen0PromotedBytesPerSec uint32

	//
	Gen1heapsize uint32

	//
	Gen1PromotedBytesPerSec uint32

	//
	Gen2heapsize uint32

	//
	LargeObjectHeapsize uint32

	//
	NumberBytesinallHeaps uint32

	//
	NumberGCHandles uint32

	//
	NumberGen0Collections uint32

	//
	NumberGen1Collections uint32

	//
	NumberGen2Collections uint32

	//
	NumberInducedGC uint32

	//
	NumberofPinnedObjects uint32

	//
	NumberofSinkBlocksinuse uint32

	//
	NumberTotalcommittedBytes uint32

	//
	NumberTotalreservedBytes uint32

	//
	PercentTimeinGC uint32

	//
	ProcessID uint32

	//
	PromotedFinalizationMemoryfromGen0 uint32

	//
	PromotedMemoryfromGen0 uint32

	//
	PromotedMemoryfromGen1 uint32
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRMemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRMemory, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRMemory{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRMemory, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRMemory{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAllocatedBytesPersec sets the value of AllocatedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyAllocatedBytesPersec(value uint32) (err error) {
	return instance.SetProperty("AllocatedBytesPersec", (value))
}

// GetAllocatedBytesPersec gets the value of AllocatedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyAllocatedBytesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AllocatedBytesPersec")
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

// SetFinalizationSurvivors sets the value of FinalizationSurvivors for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyFinalizationSurvivors(value uint32) (err error) {
	return instance.SetProperty("FinalizationSurvivors", (value))
}

// GetFinalizationSurvivors gets the value of FinalizationSurvivors for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyFinalizationSurvivors() (value uint32, err error) {
	retValue, err := instance.GetProperty("FinalizationSurvivors")
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

// SetGen0heapsize sets the value of Gen0heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyGen0heapsize(value uint32) (err error) {
	return instance.SetProperty("Gen0heapsize", (value))
}

// GetGen0heapsize gets the value of Gen0heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyGen0heapsize() (value uint32, err error) {
	retValue, err := instance.GetProperty("Gen0heapsize")
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

// SetGen0PromotedBytesPerSec sets the value of Gen0PromotedBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyGen0PromotedBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("Gen0PromotedBytesPerSec", (value))
}

// GetGen0PromotedBytesPerSec gets the value of Gen0PromotedBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyGen0PromotedBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("Gen0PromotedBytesPerSec")
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

// SetGen1heapsize sets the value of Gen1heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyGen1heapsize(value uint32) (err error) {
	return instance.SetProperty("Gen1heapsize", (value))
}

// GetGen1heapsize gets the value of Gen1heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyGen1heapsize() (value uint32, err error) {
	retValue, err := instance.GetProperty("Gen1heapsize")
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

// SetGen1PromotedBytesPerSec sets the value of Gen1PromotedBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyGen1PromotedBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("Gen1PromotedBytesPerSec", (value))
}

// GetGen1PromotedBytesPerSec gets the value of Gen1PromotedBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyGen1PromotedBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("Gen1PromotedBytesPerSec")
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

// SetGen2heapsize sets the value of Gen2heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyGen2heapsize(value uint32) (err error) {
	return instance.SetProperty("Gen2heapsize", (value))
}

// GetGen2heapsize gets the value of Gen2heapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyGen2heapsize() (value uint32, err error) {
	retValue, err := instance.GetProperty("Gen2heapsize")
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

// SetLargeObjectHeapsize sets the value of LargeObjectHeapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyLargeObjectHeapsize(value uint32) (err error) {
	return instance.SetProperty("LargeObjectHeapsize", (value))
}

// GetLargeObjectHeapsize gets the value of LargeObjectHeapsize for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyLargeObjectHeapsize() (value uint32, err error) {
	retValue, err := instance.GetProperty("LargeObjectHeapsize")
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

// SetNumberBytesinallHeaps sets the value of NumberBytesinallHeaps for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberBytesinallHeaps(value uint32) (err error) {
	return instance.SetProperty("NumberBytesinallHeaps", (value))
}

// GetNumberBytesinallHeaps gets the value of NumberBytesinallHeaps for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberBytesinallHeaps() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberBytesinallHeaps")
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

// SetNumberGCHandles sets the value of NumberGCHandles for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberGCHandles(value uint32) (err error) {
	return instance.SetProperty("NumberGCHandles", (value))
}

// GetNumberGCHandles gets the value of NumberGCHandles for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberGCHandles() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberGCHandles")
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

// SetNumberGen0Collections sets the value of NumberGen0Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberGen0Collections(value uint32) (err error) {
	return instance.SetProperty("NumberGen0Collections", (value))
}

// GetNumberGen0Collections gets the value of NumberGen0Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberGen0Collections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberGen0Collections")
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

// SetNumberGen1Collections sets the value of NumberGen1Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberGen1Collections(value uint32) (err error) {
	return instance.SetProperty("NumberGen1Collections", (value))
}

// GetNumberGen1Collections gets the value of NumberGen1Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberGen1Collections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberGen1Collections")
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

// SetNumberGen2Collections sets the value of NumberGen2Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberGen2Collections(value uint32) (err error) {
	return instance.SetProperty("NumberGen2Collections", (value))
}

// GetNumberGen2Collections gets the value of NumberGen2Collections for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberGen2Collections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberGen2Collections")
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

// SetNumberInducedGC sets the value of NumberInducedGC for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberInducedGC(value uint32) (err error) {
	return instance.SetProperty("NumberInducedGC", (value))
}

// GetNumberInducedGC gets the value of NumberInducedGC for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberInducedGC() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberInducedGC")
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

// SetNumberofPinnedObjects sets the value of NumberofPinnedObjects for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberofPinnedObjects(value uint32) (err error) {
	return instance.SetProperty("NumberofPinnedObjects", (value))
}

// GetNumberofPinnedObjects gets the value of NumberofPinnedObjects for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberofPinnedObjects() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofPinnedObjects")
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

// SetNumberofSinkBlocksinuse sets the value of NumberofSinkBlocksinuse for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberofSinkBlocksinuse(value uint32) (err error) {
	return instance.SetProperty("NumberofSinkBlocksinuse", (value))
}

// GetNumberofSinkBlocksinuse gets the value of NumberofSinkBlocksinuse for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberofSinkBlocksinuse() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofSinkBlocksinuse")
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

// SetNumberTotalcommittedBytes sets the value of NumberTotalcommittedBytes for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberTotalcommittedBytes(value uint32) (err error) {
	return instance.SetProperty("NumberTotalcommittedBytes", (value))
}

// GetNumberTotalcommittedBytes gets the value of NumberTotalcommittedBytes for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberTotalcommittedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberTotalcommittedBytes")
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

// SetNumberTotalreservedBytes sets the value of NumberTotalreservedBytes for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyNumberTotalreservedBytes(value uint32) (err error) {
	return instance.SetProperty("NumberTotalreservedBytes", (value))
}

// GetNumberTotalreservedBytes gets the value of NumberTotalreservedBytes for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyNumberTotalreservedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberTotalreservedBytes")
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

// SetPercentTimeinGC sets the value of PercentTimeinGC for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyPercentTimeinGC(value uint32) (err error) {
	return instance.SetProperty("PercentTimeinGC", (value))
}

// GetPercentTimeinGC gets the value of PercentTimeinGC for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyPercentTimeinGC() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentTimeinGC")
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

// SetProcessID sets the value of ProcessID for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyProcessID(value uint32) (err error) {
	return instance.SetProperty("ProcessID", (value))
}

// GetProcessID gets the value of ProcessID for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyProcessID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessID")
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

// SetPromotedFinalizationMemoryfromGen0 sets the value of PromotedFinalizationMemoryfromGen0 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyPromotedFinalizationMemoryfromGen0(value uint32) (err error) {
	return instance.SetProperty("PromotedFinalizationMemoryfromGen0", (value))
}

// GetPromotedFinalizationMemoryfromGen0 gets the value of PromotedFinalizationMemoryfromGen0 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyPromotedFinalizationMemoryfromGen0() (value uint32, err error) {
	retValue, err := instance.GetProperty("PromotedFinalizationMemoryfromGen0")
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

// SetPromotedMemoryfromGen0 sets the value of PromotedMemoryfromGen0 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyPromotedMemoryfromGen0(value uint32) (err error) {
	return instance.SetProperty("PromotedMemoryfromGen0", (value))
}

// GetPromotedMemoryfromGen0 gets the value of PromotedMemoryfromGen0 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyPromotedMemoryfromGen0() (value uint32, err error) {
	retValue, err := instance.GetProperty("PromotedMemoryfromGen0")
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

// SetPromotedMemoryfromGen1 sets the value of PromotedMemoryfromGen1 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) SetPropertyPromotedMemoryfromGen1(value uint32) (err error) {
	return instance.SetProperty("PromotedMemoryfromGen1", (value))
}

// GetPromotedMemoryfromGen1 gets the value of PromotedMemoryfromGen1 for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRMemory) GetPropertyPromotedMemoryfromGen1() (value uint32, err error) {
	retValue, err := instance.GetProperty("PromotedMemoryfromGen1")
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
