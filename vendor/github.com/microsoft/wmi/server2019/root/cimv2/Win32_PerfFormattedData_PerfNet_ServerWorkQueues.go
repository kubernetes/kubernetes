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

// Win32_PerfFormattedData_PerfNet_ServerWorkQueues struct
type Win32_PerfFormattedData_PerfNet_ServerWorkQueues struct {
	*Win32_PerfFormattedData

	//
	ActiveThreads uint32

	//
	AvailableThreads uint32

	//
	AvailableWorkItems uint32

	//
	BorrowedWorkItems uint32

	//
	BytesReceivedPersec uint64

	//
	BytesSentPersec uint64

	//
	BytesTransferredPersec uint64

	//
	ContextBlocksQueuedPersec uint32

	//
	CurrentClients uint32

	//
	QueueLength uint32

	//
	ReadBytesPersec uint64

	//
	ReadOperationsPersec uint64

	//
	TotalBytesPersec uint64

	//
	TotalOperationsPersec uint64

	//
	WorkItemShortages uint32

	//
	WriteBytesPersec uint64

	//
	WriteOperationsPersec uint64
}

func NewWin32_PerfFormattedData_PerfNet_ServerWorkQueuesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfNet_ServerWorkQueues{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfNet_ServerWorkQueuesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfNet_ServerWorkQueues{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetActiveThreads sets the value of ActiveThreads for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyActiveThreads(value uint32) (err error) {
	return instance.SetProperty("ActiveThreads", (value))
}

// GetActiveThreads gets the value of ActiveThreads for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyActiveThreads() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveThreads")
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

// SetAvailableThreads sets the value of AvailableThreads for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyAvailableThreads(value uint32) (err error) {
	return instance.SetProperty("AvailableThreads", (value))
}

// GetAvailableThreads gets the value of AvailableThreads for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyAvailableThreads() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvailableThreads")
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

// SetAvailableWorkItems sets the value of AvailableWorkItems for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyAvailableWorkItems(value uint32) (err error) {
	return instance.SetProperty("AvailableWorkItems", (value))
}

// GetAvailableWorkItems gets the value of AvailableWorkItems for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyAvailableWorkItems() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvailableWorkItems")
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

// SetBorrowedWorkItems sets the value of BorrowedWorkItems for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyBorrowedWorkItems(value uint32) (err error) {
	return instance.SetProperty("BorrowedWorkItems", (value))
}

// GetBorrowedWorkItems gets the value of BorrowedWorkItems for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyBorrowedWorkItems() (value uint32, err error) {
	retValue, err := instance.GetProperty("BorrowedWorkItems")
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec")
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

// SetBytesSentPersec sets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSentPersec")
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

// SetBytesTransferredPersec sets the value of BytesTransferredPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyBytesTransferredPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTransferredPersec", (value))
}

// GetBytesTransferredPersec gets the value of BytesTransferredPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyBytesTransferredPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransferredPersec")
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

// SetContextBlocksQueuedPersec sets the value of ContextBlocksQueuedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyContextBlocksQueuedPersec(value uint32) (err error) {
	return instance.SetProperty("ContextBlocksQueuedPersec", (value))
}

// GetContextBlocksQueuedPersec gets the value of ContextBlocksQueuedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyContextBlocksQueuedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ContextBlocksQueuedPersec")
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

// SetCurrentClients sets the value of CurrentClients for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyCurrentClients(value uint32) (err error) {
	return instance.SetProperty("CurrentClients", (value))
}

// GetCurrentClients gets the value of CurrentClients for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyCurrentClients() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentClients")
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

// SetQueueLength sets the value of QueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyQueueLength(value uint32) (err error) {
	return instance.SetProperty("QueueLength", (value))
}

// GetQueueLength gets the value of QueueLength for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyQueueLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueueLength")
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

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPersec")
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

// SetReadOperationsPersec sets the value of ReadOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyReadOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadOperationsPersec", (value))
}

// GetReadOperationsPersec gets the value of ReadOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyReadOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadOperationsPersec")
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

// SetTotalBytesPersec sets the value of TotalBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyTotalBytesPersec(value uint64) (err error) {
	return instance.SetProperty("TotalBytesPersec", (value))
}

// GetTotalBytesPersec gets the value of TotalBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyTotalBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalBytesPersec")
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

// SetTotalOperationsPersec sets the value of TotalOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyTotalOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("TotalOperationsPersec", (value))
}

// GetTotalOperationsPersec gets the value of TotalOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyTotalOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOperationsPersec")
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

// SetWorkItemShortages sets the value of WorkItemShortages for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyWorkItemShortages(value uint32) (err error) {
	return instance.SetProperty("WorkItemShortages", (value))
}

// GetWorkItemShortages gets the value of WorkItemShortages for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyWorkItemShortages() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkItemShortages")
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

// SetWriteBytesPersec sets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersec")
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

// SetWriteOperationsPersec sets the value of WriteOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) SetPropertyWriteOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("WriteOperationsPersec", (value))
}

// GetWriteOperationsPersec gets the value of WriteOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_ServerWorkQueues) GetPropertyWriteOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteOperationsPersec")
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
