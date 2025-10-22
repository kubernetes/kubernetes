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

// Win32_Process struct
type Win32_Process struct {
	*CIM_Process

	//
	CommandLine string

	//
	ExecutablePath string

	//
	HandleCount uint32

	//
	MaximumWorkingSetSize uint32

	//
	MinimumWorkingSetSize uint32

	//
	OtherOperationCount uint64

	//
	OtherTransferCount uint64

	//
	PageFaults uint32

	//
	PageFileUsage uint32

	//
	ParentProcessId uint32

	//
	PeakPageFileUsage uint32

	//
	PeakVirtualSize uint64

	//
	PeakWorkingSetSize uint32

	//
	PrivatePageCount uint64

	//
	ProcessId uint32

	//
	QuotaNonPagedPoolUsage uint32

	//
	QuotaPagedPoolUsage uint32

	//
	QuotaPeakNonPagedPoolUsage uint32

	//
	QuotaPeakPagedPoolUsage uint32

	//
	ReadOperationCount uint64

	//
	ReadTransferCount uint64

	//
	SessionId uint32

	//
	ThreadCount uint32

	//
	VirtualSize uint64

	//
	WindowsVersion string

	//
	WriteOperationCount uint64

	//
	WriteTransferCount uint64
}

func NewWin32_ProcessEx1(instance *cim.WmiInstance) (newInstance *Win32_Process, err error) {
	tmp, err := NewCIM_ProcessEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Process{
		CIM_Process: tmp,
	}
	return
}

func NewWin32_ProcessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Process, err error) {
	tmp, err := NewCIM_ProcessEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Process{
		CIM_Process: tmp,
	}
	return
}

// SetCommandLine sets the value of CommandLine for the instance
func (instance *Win32_Process) SetPropertyCommandLine(value string) (err error) {
	return instance.SetProperty("CommandLine", (value))
}

// GetCommandLine gets the value of CommandLine for the instance
func (instance *Win32_Process) GetPropertyCommandLine() (value string, err error) {
	retValue, err := instance.GetProperty("CommandLine")
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

// SetExecutablePath sets the value of ExecutablePath for the instance
func (instance *Win32_Process) SetPropertyExecutablePath(value string) (err error) {
	return instance.SetProperty("ExecutablePath", (value))
}

// GetExecutablePath gets the value of ExecutablePath for the instance
func (instance *Win32_Process) GetPropertyExecutablePath() (value string, err error) {
	retValue, err := instance.GetProperty("ExecutablePath")
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

// SetHandleCount sets the value of HandleCount for the instance
func (instance *Win32_Process) SetPropertyHandleCount(value uint32) (err error) {
	return instance.SetProperty("HandleCount", (value))
}

// GetHandleCount gets the value of HandleCount for the instance
func (instance *Win32_Process) GetPropertyHandleCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("HandleCount")
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

// SetMaximumWorkingSetSize sets the value of MaximumWorkingSetSize for the instance
func (instance *Win32_Process) SetPropertyMaximumWorkingSetSize(value uint32) (err error) {
	return instance.SetProperty("MaximumWorkingSetSize", (value))
}

// GetMaximumWorkingSetSize gets the value of MaximumWorkingSetSize for the instance
func (instance *Win32_Process) GetPropertyMaximumWorkingSetSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumWorkingSetSize")
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

// SetMinimumWorkingSetSize sets the value of MinimumWorkingSetSize for the instance
func (instance *Win32_Process) SetPropertyMinimumWorkingSetSize(value uint32) (err error) {
	return instance.SetProperty("MinimumWorkingSetSize", (value))
}

// GetMinimumWorkingSetSize gets the value of MinimumWorkingSetSize for the instance
func (instance *Win32_Process) GetPropertyMinimumWorkingSetSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinimumWorkingSetSize")
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

// SetOtherOperationCount sets the value of OtherOperationCount for the instance
func (instance *Win32_Process) SetPropertyOtherOperationCount(value uint64) (err error) {
	return instance.SetProperty("OtherOperationCount", (value))
}

// GetOtherOperationCount gets the value of OtherOperationCount for the instance
func (instance *Win32_Process) GetPropertyOtherOperationCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("OtherOperationCount")
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

// SetOtherTransferCount sets the value of OtherTransferCount for the instance
func (instance *Win32_Process) SetPropertyOtherTransferCount(value uint64) (err error) {
	return instance.SetProperty("OtherTransferCount", (value))
}

// GetOtherTransferCount gets the value of OtherTransferCount for the instance
func (instance *Win32_Process) GetPropertyOtherTransferCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("OtherTransferCount")
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

// SetPageFaults sets the value of PageFaults for the instance
func (instance *Win32_Process) SetPropertyPageFaults(value uint32) (err error) {
	return instance.SetProperty("PageFaults", (value))
}

// GetPageFaults gets the value of PageFaults for the instance
func (instance *Win32_Process) GetPropertyPageFaults() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageFaults")
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

// SetPageFileUsage sets the value of PageFileUsage for the instance
func (instance *Win32_Process) SetPropertyPageFileUsage(value uint32) (err error) {
	return instance.SetProperty("PageFileUsage", (value))
}

// GetPageFileUsage gets the value of PageFileUsage for the instance
func (instance *Win32_Process) GetPropertyPageFileUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageFileUsage")
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

// SetParentProcessId sets the value of ParentProcessId for the instance
func (instance *Win32_Process) SetPropertyParentProcessId(value uint32) (err error) {
	return instance.SetProperty("ParentProcessId", (value))
}

// GetParentProcessId gets the value of ParentProcessId for the instance
func (instance *Win32_Process) GetPropertyParentProcessId() (value uint32, err error) {
	retValue, err := instance.GetProperty("ParentProcessId")
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

// SetPeakPageFileUsage sets the value of PeakPageFileUsage for the instance
func (instance *Win32_Process) SetPropertyPeakPageFileUsage(value uint32) (err error) {
	return instance.SetProperty("PeakPageFileUsage", (value))
}

// GetPeakPageFileUsage gets the value of PeakPageFileUsage for the instance
func (instance *Win32_Process) GetPropertyPeakPageFileUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("PeakPageFileUsage")
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

// SetPeakVirtualSize sets the value of PeakVirtualSize for the instance
func (instance *Win32_Process) SetPropertyPeakVirtualSize(value uint64) (err error) {
	return instance.SetProperty("PeakVirtualSize", (value))
}

// GetPeakVirtualSize gets the value of PeakVirtualSize for the instance
func (instance *Win32_Process) GetPropertyPeakVirtualSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("PeakVirtualSize")
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

// SetPeakWorkingSetSize sets the value of PeakWorkingSetSize for the instance
func (instance *Win32_Process) SetPropertyPeakWorkingSetSize(value uint32) (err error) {
	return instance.SetProperty("PeakWorkingSetSize", (value))
}

// GetPeakWorkingSetSize gets the value of PeakWorkingSetSize for the instance
func (instance *Win32_Process) GetPropertyPeakWorkingSetSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("PeakWorkingSetSize")
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

// SetPrivatePageCount sets the value of PrivatePageCount for the instance
func (instance *Win32_Process) SetPropertyPrivatePageCount(value uint64) (err error) {
	return instance.SetProperty("PrivatePageCount", (value))
}

// GetPrivatePageCount gets the value of PrivatePageCount for the instance
func (instance *Win32_Process) GetPropertyPrivatePageCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("PrivatePageCount")
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

// SetProcessId sets the value of ProcessId for the instance
func (instance *Win32_Process) SetPropertyProcessId(value uint32) (err error) {
	return instance.SetProperty("ProcessId", (value))
}

// GetProcessId gets the value of ProcessId for the instance
func (instance *Win32_Process) GetPropertyProcessId() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessId")
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

// SetQuotaNonPagedPoolUsage sets the value of QuotaNonPagedPoolUsage for the instance
func (instance *Win32_Process) SetPropertyQuotaNonPagedPoolUsage(value uint32) (err error) {
	return instance.SetProperty("QuotaNonPagedPoolUsage", (value))
}

// GetQuotaNonPagedPoolUsage gets the value of QuotaNonPagedPoolUsage for the instance
func (instance *Win32_Process) GetPropertyQuotaNonPagedPoolUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuotaNonPagedPoolUsage")
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

// SetQuotaPagedPoolUsage sets the value of QuotaPagedPoolUsage for the instance
func (instance *Win32_Process) SetPropertyQuotaPagedPoolUsage(value uint32) (err error) {
	return instance.SetProperty("QuotaPagedPoolUsage", (value))
}

// GetQuotaPagedPoolUsage gets the value of QuotaPagedPoolUsage for the instance
func (instance *Win32_Process) GetPropertyQuotaPagedPoolUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuotaPagedPoolUsage")
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

// SetQuotaPeakNonPagedPoolUsage sets the value of QuotaPeakNonPagedPoolUsage for the instance
func (instance *Win32_Process) SetPropertyQuotaPeakNonPagedPoolUsage(value uint32) (err error) {
	return instance.SetProperty("QuotaPeakNonPagedPoolUsage", (value))
}

// GetQuotaPeakNonPagedPoolUsage gets the value of QuotaPeakNonPagedPoolUsage for the instance
func (instance *Win32_Process) GetPropertyQuotaPeakNonPagedPoolUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuotaPeakNonPagedPoolUsage")
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

// SetQuotaPeakPagedPoolUsage sets the value of QuotaPeakPagedPoolUsage for the instance
func (instance *Win32_Process) SetPropertyQuotaPeakPagedPoolUsage(value uint32) (err error) {
	return instance.SetProperty("QuotaPeakPagedPoolUsage", (value))
}

// GetQuotaPeakPagedPoolUsage gets the value of QuotaPeakPagedPoolUsage for the instance
func (instance *Win32_Process) GetPropertyQuotaPeakPagedPoolUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuotaPeakPagedPoolUsage")
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

// SetReadOperationCount sets the value of ReadOperationCount for the instance
func (instance *Win32_Process) SetPropertyReadOperationCount(value uint64) (err error) {
	return instance.SetProperty("ReadOperationCount", (value))
}

// GetReadOperationCount gets the value of ReadOperationCount for the instance
func (instance *Win32_Process) GetPropertyReadOperationCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadOperationCount")
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

// SetReadTransferCount sets the value of ReadTransferCount for the instance
func (instance *Win32_Process) SetPropertyReadTransferCount(value uint64) (err error) {
	return instance.SetProperty("ReadTransferCount", (value))
}

// GetReadTransferCount gets the value of ReadTransferCount for the instance
func (instance *Win32_Process) GetPropertyReadTransferCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadTransferCount")
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

// SetSessionId sets the value of SessionId for the instance
func (instance *Win32_Process) SetPropertySessionId(value uint32) (err error) {
	return instance.SetProperty("SessionId", (value))
}

// GetSessionId gets the value of SessionId for the instance
func (instance *Win32_Process) GetPropertySessionId() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionId")
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

// SetThreadCount sets the value of ThreadCount for the instance
func (instance *Win32_Process) SetPropertyThreadCount(value uint32) (err error) {
	return instance.SetProperty("ThreadCount", (value))
}

// GetThreadCount gets the value of ThreadCount for the instance
func (instance *Win32_Process) GetPropertyThreadCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadCount")
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

// SetVirtualSize sets the value of VirtualSize for the instance
func (instance *Win32_Process) SetPropertyVirtualSize(value uint64) (err error) {
	return instance.SetProperty("VirtualSize", (value))
}

// GetVirtualSize gets the value of VirtualSize for the instance
func (instance *Win32_Process) GetPropertyVirtualSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualSize")
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

// SetWindowsVersion sets the value of WindowsVersion for the instance
func (instance *Win32_Process) SetPropertyWindowsVersion(value string) (err error) {
	return instance.SetProperty("WindowsVersion", (value))
}

// GetWindowsVersion gets the value of WindowsVersion for the instance
func (instance *Win32_Process) GetPropertyWindowsVersion() (value string, err error) {
	retValue, err := instance.GetProperty("WindowsVersion")
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

// SetWriteOperationCount sets the value of WriteOperationCount for the instance
func (instance *Win32_Process) SetPropertyWriteOperationCount(value uint64) (err error) {
	return instance.SetProperty("WriteOperationCount", (value))
}

// GetWriteOperationCount gets the value of WriteOperationCount for the instance
func (instance *Win32_Process) GetPropertyWriteOperationCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteOperationCount")
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

// SetWriteTransferCount sets the value of WriteTransferCount for the instance
func (instance *Win32_Process) SetPropertyWriteTransferCount(value uint64) (err error) {
	return instance.SetProperty("WriteTransferCount", (value))
}

// GetWriteTransferCount gets the value of WriteTransferCount for the instance
func (instance *Win32_Process) GetPropertyWriteTransferCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteTransferCount")
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

// <param name="CommandLine" type="string "></param>
// <param name="CurrentDirectory" type="string "></param>
// <param name="ProcessStartupInformation" type="Win32_ProcessStartup "></param>

// <param name="ProcessId" type="uint32 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Process) Create( /* IN */ CommandLine string,
	/* IN */ CurrentDirectory string,
	/* IN */ ProcessStartupInformation Win32_ProcessStartup,
	/* OUT */ ProcessId uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Create", CommandLine, CurrentDirectory, ProcessStartupInformation)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Reason" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Process) Terminate( /* IN */ Reason uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Terminate", Reason)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Domain" type="string "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="User" type="string "></param>
func (instance *Win32_Process) GetOwner( /* OUT */ User string,
	/* OUT */ Domain string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetOwner")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="Sid" type="string "></param>
func (instance *Win32_Process) GetOwnerSid( /* OUT */ Sid string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetOwnerSid")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Priority" type="int32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Process) SetPriority( /* IN */ Priority int32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPriority", Priority)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Process) AttachDebugger() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AttachDebugger")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AvailableVirtualSize" type="uint64 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Process) GetAvailableVirtualSize( /* OUT */ AvailableVirtualSize uint64) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetAvailableVirtualSize")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
