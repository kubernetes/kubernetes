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

// Win32_PerfFormattedData_PerfNet_Server struct
type Win32_PerfFormattedData_PerfNet_Server struct {
	*Win32_PerfFormattedData

	//
	BlockingRequestsRejected uint32

	//
	BytesReceivedPersec uint64

	//
	BytesTotalPersec uint64

	//
	BytesTransmittedPersec uint64

	//
	ContextBlocksQueuedPersec uint32

	//
	ErrorsAccessPermissions uint32

	//
	ErrorsGrantedAccess uint32

	//
	ErrorsLogon uint32

	//
	ErrorsSystem uint32

	//
	FileDirectorySearches uint32

	//
	FilesOpen uint32

	//
	FilesOpenedTotal uint32

	//
	LogonPersec uint32

	//
	LogonTotal uint32

	//
	PoolNonpagedBytes uint32

	//
	PoolNonpagedFailures uint32

	//
	PoolNonpagedPeak uint32

	//
	PoolPagedBytes uint32

	//
	PoolPagedFailures uint32

	//
	PoolPagedPeak uint32

	//
	ReconnectedDurableHandles uint32

	//
	ReconnectedResilientHandles uint32

	//
	ServerSessions uint32

	//
	SessionsErroredOut uint32

	//
	SessionsForcedOff uint32

	//
	SessionsLoggedOff uint32

	//
	SessionsTimedOut uint32

	//
	SMBBranchCacheHashBytesSent uint64

	//
	SMBBranchCacheHashGenerationRequests uint32

	//
	SMBBranchCacheHashHeaderRequests uint32

	//
	SMBBranchCacheHashRequestsReceived uint32

	//
	SMBBranchCacheHashResponsesSent uint32

	//
	SMBBranchCacheHashV2BytesSent uint64

	//
	SMBBranchCacheHashV2GenerationRequests uint32

	//
	SMBBranchCacheHashV2HeaderRequests uint32

	//
	SMBBranchCacheHashV2RequestsReceived uint32

	//
	SMBBranchCacheHashV2RequestsServedFromDedup uint32

	//
	SMBBranchCacheHashV2ResponsesSent uint32

	//
	TotalDurableHandles uint32

	//
	TotalResilientHandles uint32

	//
	WorkItemShortages uint32
}

func NewWin32_PerfFormattedData_PerfNet_ServerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfNet_Server, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfNet_Server{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfNet_ServerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfNet_Server, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfNet_Server{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBlockingRequestsRejected sets the value of BlockingRequestsRejected for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyBlockingRequestsRejected(value uint32) (err error) {
	return instance.SetProperty("BlockingRequestsRejected", (value))
}

// GetBlockingRequestsRejected gets the value of BlockingRequestsRejected for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyBlockingRequestsRejected() (value uint32, err error) {
	retValue, err := instance.GetProperty("BlockingRequestsRejected")
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
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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

// SetBytesTotalPersec sets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyBytesTotalPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTotalPersec", (value))
}

// GetBytesTotalPersec gets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyBytesTotalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTotalPersec")
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

// SetBytesTransmittedPersec sets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyBytesTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTransmittedPersec", (value))
}

// GetBytesTransmittedPersec gets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyBytesTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPersec")
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
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyContextBlocksQueuedPersec(value uint32) (err error) {
	return instance.SetProperty("ContextBlocksQueuedPersec", (value))
}

// GetContextBlocksQueuedPersec gets the value of ContextBlocksQueuedPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyContextBlocksQueuedPersec() (value uint32, err error) {
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

// SetErrorsAccessPermissions sets the value of ErrorsAccessPermissions for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyErrorsAccessPermissions(value uint32) (err error) {
	return instance.SetProperty("ErrorsAccessPermissions", (value))
}

// GetErrorsAccessPermissions gets the value of ErrorsAccessPermissions for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyErrorsAccessPermissions() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorsAccessPermissions")
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

// SetErrorsGrantedAccess sets the value of ErrorsGrantedAccess for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyErrorsGrantedAccess(value uint32) (err error) {
	return instance.SetProperty("ErrorsGrantedAccess", (value))
}

// GetErrorsGrantedAccess gets the value of ErrorsGrantedAccess for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyErrorsGrantedAccess() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorsGrantedAccess")
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

// SetErrorsLogon sets the value of ErrorsLogon for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyErrorsLogon(value uint32) (err error) {
	return instance.SetProperty("ErrorsLogon", (value))
}

// GetErrorsLogon gets the value of ErrorsLogon for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyErrorsLogon() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorsLogon")
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

// SetErrorsSystem sets the value of ErrorsSystem for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyErrorsSystem(value uint32) (err error) {
	return instance.SetProperty("ErrorsSystem", (value))
}

// GetErrorsSystem gets the value of ErrorsSystem for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyErrorsSystem() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorsSystem")
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

// SetFileDirectorySearches sets the value of FileDirectorySearches for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyFileDirectorySearches(value uint32) (err error) {
	return instance.SetProperty("FileDirectorySearches", (value))
}

// GetFileDirectorySearches gets the value of FileDirectorySearches for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyFileDirectorySearches() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileDirectorySearches")
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

// SetFilesOpen sets the value of FilesOpen for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyFilesOpen(value uint32) (err error) {
	return instance.SetProperty("FilesOpen", (value))
}

// GetFilesOpen gets the value of FilesOpen for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyFilesOpen() (value uint32, err error) {
	retValue, err := instance.GetProperty("FilesOpen")
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

// SetFilesOpenedTotal sets the value of FilesOpenedTotal for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyFilesOpenedTotal(value uint32) (err error) {
	return instance.SetProperty("FilesOpenedTotal", (value))
}

// GetFilesOpenedTotal gets the value of FilesOpenedTotal for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyFilesOpenedTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("FilesOpenedTotal")
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

// SetLogonPersec sets the value of LogonPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyLogonPersec(value uint32) (err error) {
	return instance.SetProperty("LogonPersec", (value))
}

// GetLogonPersec gets the value of LogonPersec for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyLogonPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogonPersec")
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

// SetLogonTotal sets the value of LogonTotal for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyLogonTotal(value uint32) (err error) {
	return instance.SetProperty("LogonTotal", (value))
}

// GetLogonTotal gets the value of LogonTotal for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyLogonTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogonTotal")
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

// SetPoolNonpagedBytes sets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolNonpagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedBytes", (value))
}

// GetPoolNonpagedBytes gets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolNonpagedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedBytes")
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

// SetPoolNonpagedFailures sets the value of PoolNonpagedFailures for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolNonpagedFailures(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedFailures", (value))
}

// GetPoolNonpagedFailures gets the value of PoolNonpagedFailures for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolNonpagedFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedFailures")
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

// SetPoolNonpagedPeak sets the value of PoolNonpagedPeak for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolNonpagedPeak(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedPeak", (value))
}

// GetPoolNonpagedPeak gets the value of PoolNonpagedPeak for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolNonpagedPeak() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedPeak")
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

// SetPoolPagedBytes sets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolPagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolPagedBytes", (value))
}

// GetPoolPagedBytes gets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolPagedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolPagedBytes")
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

// SetPoolPagedFailures sets the value of PoolPagedFailures for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolPagedFailures(value uint32) (err error) {
	return instance.SetProperty("PoolPagedFailures", (value))
}

// GetPoolPagedFailures gets the value of PoolPagedFailures for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolPagedFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolPagedFailures")
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

// SetPoolPagedPeak sets the value of PoolPagedPeak for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyPoolPagedPeak(value uint32) (err error) {
	return instance.SetProperty("PoolPagedPeak", (value))
}

// GetPoolPagedPeak gets the value of PoolPagedPeak for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyPoolPagedPeak() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolPagedPeak")
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

// SetReconnectedDurableHandles sets the value of ReconnectedDurableHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyReconnectedDurableHandles(value uint32) (err error) {
	return instance.SetProperty("ReconnectedDurableHandles", (value))
}

// GetReconnectedDurableHandles gets the value of ReconnectedDurableHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyReconnectedDurableHandles() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReconnectedDurableHandles")
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

// SetReconnectedResilientHandles sets the value of ReconnectedResilientHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyReconnectedResilientHandles(value uint32) (err error) {
	return instance.SetProperty("ReconnectedResilientHandles", (value))
}

// GetReconnectedResilientHandles gets the value of ReconnectedResilientHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyReconnectedResilientHandles() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReconnectedResilientHandles")
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

// SetServerSessions sets the value of ServerSessions for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyServerSessions(value uint32) (err error) {
	return instance.SetProperty("ServerSessions", (value))
}

// GetServerSessions gets the value of ServerSessions for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyServerSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServerSessions")
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

// SetSessionsErroredOut sets the value of SessionsErroredOut for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySessionsErroredOut(value uint32) (err error) {
	return instance.SetProperty("SessionsErroredOut", (value))
}

// GetSessionsErroredOut gets the value of SessionsErroredOut for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySessionsErroredOut() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsErroredOut")
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

// SetSessionsForcedOff sets the value of SessionsForcedOff for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySessionsForcedOff(value uint32) (err error) {
	return instance.SetProperty("SessionsForcedOff", (value))
}

// GetSessionsForcedOff gets the value of SessionsForcedOff for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySessionsForcedOff() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsForcedOff")
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

// SetSessionsLoggedOff sets the value of SessionsLoggedOff for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySessionsLoggedOff(value uint32) (err error) {
	return instance.SetProperty("SessionsLoggedOff", (value))
}

// GetSessionsLoggedOff gets the value of SessionsLoggedOff for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySessionsLoggedOff() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsLoggedOff")
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

// SetSessionsTimedOut sets the value of SessionsTimedOut for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySessionsTimedOut(value uint32) (err error) {
	return instance.SetProperty("SessionsTimedOut", (value))
}

// GetSessionsTimedOut gets the value of SessionsTimedOut for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySessionsTimedOut() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsTimedOut")
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

// SetSMBBranchCacheHashBytesSent sets the value of SMBBranchCacheHashBytesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashBytesSent(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheHashBytesSent", (value))
}

// GetSMBBranchCacheHashBytesSent gets the value of SMBBranchCacheHashBytesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashBytesSent")
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

// SetSMBBranchCacheHashGenerationRequests sets the value of SMBBranchCacheHashGenerationRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashGenerationRequests(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashGenerationRequests", (value))
}

// GetSMBBranchCacheHashGenerationRequests gets the value of SMBBranchCacheHashGenerationRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashGenerationRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashGenerationRequests")
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

// SetSMBBranchCacheHashHeaderRequests sets the value of SMBBranchCacheHashHeaderRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashHeaderRequests(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashHeaderRequests", (value))
}

// GetSMBBranchCacheHashHeaderRequests gets the value of SMBBranchCacheHashHeaderRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashHeaderRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashHeaderRequests")
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

// SetSMBBranchCacheHashRequestsReceived sets the value of SMBBranchCacheHashRequestsReceived for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashRequestsReceived(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashRequestsReceived", (value))
}

// GetSMBBranchCacheHashRequestsReceived gets the value of SMBBranchCacheHashRequestsReceived for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashRequestsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashRequestsReceived")
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

// SetSMBBranchCacheHashResponsesSent sets the value of SMBBranchCacheHashResponsesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashResponsesSent(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashResponsesSent", (value))
}

// GetSMBBranchCacheHashResponsesSent gets the value of SMBBranchCacheHashResponsesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashResponsesSent() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashResponsesSent")
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

// SetSMBBranchCacheHashV2BytesSent sets the value of SMBBranchCacheHashV2BytesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2BytesSent(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2BytesSent", (value))
}

// GetSMBBranchCacheHashV2BytesSent gets the value of SMBBranchCacheHashV2BytesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2BytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2BytesSent")
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

// SetSMBBranchCacheHashV2GenerationRequests sets the value of SMBBranchCacheHashV2GenerationRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2GenerationRequests(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2GenerationRequests", (value))
}

// GetSMBBranchCacheHashV2GenerationRequests gets the value of SMBBranchCacheHashV2GenerationRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2GenerationRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2GenerationRequests")
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

// SetSMBBranchCacheHashV2HeaderRequests sets the value of SMBBranchCacheHashV2HeaderRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2HeaderRequests(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2HeaderRequests", (value))
}

// GetSMBBranchCacheHashV2HeaderRequests gets the value of SMBBranchCacheHashV2HeaderRequests for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2HeaderRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2HeaderRequests")
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

// SetSMBBranchCacheHashV2RequestsReceived sets the value of SMBBranchCacheHashV2RequestsReceived for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2RequestsReceived(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2RequestsReceived", (value))
}

// GetSMBBranchCacheHashV2RequestsReceived gets the value of SMBBranchCacheHashV2RequestsReceived for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2RequestsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2RequestsReceived")
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

// SetSMBBranchCacheHashV2RequestsServedFromDedup sets the value of SMBBranchCacheHashV2RequestsServedFromDedup for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2RequestsServedFromDedup(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2RequestsServedFromDedup", (value))
}

// GetSMBBranchCacheHashV2RequestsServedFromDedup gets the value of SMBBranchCacheHashV2RequestsServedFromDedup for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2RequestsServedFromDedup() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2RequestsServedFromDedup")
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

// SetSMBBranchCacheHashV2ResponsesSent sets the value of SMBBranchCacheHashV2ResponsesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertySMBBranchCacheHashV2ResponsesSent(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashV2ResponsesSent", (value))
}

// GetSMBBranchCacheHashV2ResponsesSent gets the value of SMBBranchCacheHashV2ResponsesSent for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertySMBBranchCacheHashV2ResponsesSent() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashV2ResponsesSent")
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

// SetTotalDurableHandles sets the value of TotalDurableHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyTotalDurableHandles(value uint32) (err error) {
	return instance.SetProperty("TotalDurableHandles", (value))
}

// GetTotalDurableHandles gets the value of TotalDurableHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyTotalDurableHandles() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalDurableHandles")
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

// SetTotalResilientHandles sets the value of TotalResilientHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyTotalResilientHandles(value uint32) (err error) {
	return instance.SetProperty("TotalResilientHandles", (value))
}

// GetTotalResilientHandles gets the value of TotalResilientHandles for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyTotalResilientHandles() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalResilientHandles")
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

// SetWorkItemShortages sets the value of WorkItemShortages for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) SetPropertyWorkItemShortages(value uint32) (err error) {
	return instance.SetProperty("WorkItemShortages", (value))
}

// GetWorkItemShortages gets the value of WorkItemShortages for the instance
func (instance *Win32_PerfFormattedData_PerfNet_Server) GetPropertyWorkItemShortages() (value uint32, err error) {
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
