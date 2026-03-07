// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 3/19/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter struct
type Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter struct {
	*Win32_PerfRawData

	//
	ACKRecievedPerSec uint64

	//
	ACKSentPerSec uint64

	//
	HeartbeatsRecievedPerSec uint64

	//
	HeartbeatsSentPerSec uint64

	//
	TotalACKRecieved uint64

	//
	TotalACKSent uint64

	//
	TotalHeartbeatsRecieved uint64

	//
	TotalHeartbeatsSent uint64

	//
	TotalReceivesDropped uint64

	//
	TotalSendRequestsDropped uint64
}

func NewWin32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetACKRecievedPerSec sets the value of ACKRecievedPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyACKRecievedPerSec(value uint64) (err error) {
	return instance.SetProperty("ACKRecievedPerSec", value)
}

// GetACKRecievedPerSec gets the value of ACKRecievedPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyACKRecievedPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ACKRecievedPerSec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetACKSentPerSec sets the value of ACKSentPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyACKSentPerSec(value uint64) (err error) {
	return instance.SetProperty("ACKSentPerSec", value)
}

// GetACKSentPerSec gets the value of ACKSentPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyACKSentPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ACKSentPerSec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetHeartbeatsRecievedPerSec sets the value of HeartbeatsRecievedPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyHeartbeatsRecievedPerSec(value uint64) (err error) {
	return instance.SetProperty("HeartbeatsRecievedPerSec", value)
}

// GetHeartbeatsRecievedPerSec gets the value of HeartbeatsRecievedPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyHeartbeatsRecievedPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("HeartbeatsRecievedPerSec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetHeartbeatsSentPerSec sets the value of HeartbeatsSentPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyHeartbeatsSentPerSec(value uint64) (err error) {
	return instance.SetProperty("HeartbeatsSentPerSec", value)
}

// GetHeartbeatsSentPerSec gets the value of HeartbeatsSentPerSec for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyHeartbeatsSentPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("HeartbeatsSentPerSec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalACKRecieved sets the value of TotalACKRecieved for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalACKRecieved(value uint64) (err error) {
	return instance.SetProperty("TotalACKRecieved", value)
}

// GetTotalACKRecieved gets the value of TotalACKRecieved for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalACKRecieved() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalACKRecieved")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalACKSent sets the value of TotalACKSent for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalACKSent(value uint64) (err error) {
	return instance.SetProperty("TotalACKSent", value)
}

// GetTotalACKSent gets the value of TotalACKSent for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalACKSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalACKSent")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalHeartbeatsRecieved sets the value of TotalHeartbeatsRecieved for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalHeartbeatsRecieved(value uint64) (err error) {
	return instance.SetProperty("TotalHeartbeatsRecieved", value)
}

// GetTotalHeartbeatsRecieved gets the value of TotalHeartbeatsRecieved for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalHeartbeatsRecieved() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalHeartbeatsRecieved")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalHeartbeatsSent sets the value of TotalHeartbeatsSent for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalHeartbeatsSent(value uint64) (err error) {
	return instance.SetProperty("TotalHeartbeatsSent", value)
}

// GetTotalHeartbeatsSent gets the value of TotalHeartbeatsSent for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalHeartbeatsSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalHeartbeatsSent")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalReceivesDropped sets the value of TotalReceivesDropped for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalReceivesDropped(value uint64) (err error) {
	return instance.SetProperty("TotalReceivesDropped", value)
}

// GetTotalReceivesDropped gets the value of TotalReceivesDropped for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalReceivesDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalReceivesDropped")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetTotalSendRequestsDropped sets the value of TotalSendRequestsDropped for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) SetPropertyTotalSendRequestsDropped(value uint64) (err error) {
	return instance.SetProperty("TotalSendRequestsDropped", value)
}

// GetTotalSendRequestsDropped gets the value of TotalSendRequestsDropped for the instance
func (instance *Win32_PerfRawData_NetFtPerfProvider_ClusterNetftRouteMonitorAdapter) GetPropertyTotalSendRequestsDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalSendRequestsDropped")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}
