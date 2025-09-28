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

// Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM struct
type Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM struct {
	*Win32_PerfRawData

	//
	AverageReplicationLatency uint64

	//
	AverageReplicationSize uint64

	//
	CompressionEfficiency uint64

	//
	LastReplicationSize uint64

	//
	NetworkBytesRecv uint64

	//
	NetworkBytesSent uint64

	//
	ReplicationCount uint32

	//
	ReplicationLatency uint64

	//
	ResynchronizedBytes uint64
}

func NewWin32_PerfRawData_HyperVReplicaStats_HyperVReplicaVMEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_HyperVReplicaStats_HyperVReplicaVMEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAverageReplicationLatency sets the value of AverageReplicationLatency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyAverageReplicationLatency(value uint64) (err error) {
	return instance.SetProperty("AverageReplicationLatency", (value))
}

// GetAverageReplicationLatency gets the value of AverageReplicationLatency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyAverageReplicationLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageReplicationLatency")
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

// SetAverageReplicationSize sets the value of AverageReplicationSize for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyAverageReplicationSize(value uint64) (err error) {
	return instance.SetProperty("AverageReplicationSize", (value))
}

// GetAverageReplicationSize gets the value of AverageReplicationSize for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyAverageReplicationSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageReplicationSize")
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

// SetCompressionEfficiency sets the value of CompressionEfficiency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyCompressionEfficiency(value uint64) (err error) {
	return instance.SetProperty("CompressionEfficiency", (value))
}

// GetCompressionEfficiency gets the value of CompressionEfficiency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyCompressionEfficiency() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressionEfficiency")
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

// SetLastReplicationSize sets the value of LastReplicationSize for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyLastReplicationSize(value uint64) (err error) {
	return instance.SetProperty("LastReplicationSize", (value))
}

// GetLastReplicationSize gets the value of LastReplicationSize for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyLastReplicationSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("LastReplicationSize")
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

// SetNetworkBytesRecv sets the value of NetworkBytesRecv for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyNetworkBytesRecv(value uint64) (err error) {
	return instance.SetProperty("NetworkBytesRecv", (value))
}

// GetNetworkBytesRecv gets the value of NetworkBytesRecv for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyNetworkBytesRecv() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkBytesRecv")
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

// SetNetworkBytesSent sets the value of NetworkBytesSent for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyNetworkBytesSent(value uint64) (err error) {
	return instance.SetProperty("NetworkBytesSent", (value))
}

// GetNetworkBytesSent gets the value of NetworkBytesSent for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyNetworkBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkBytesSent")
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

// SetReplicationCount sets the value of ReplicationCount for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyReplicationCount(value uint32) (err error) {
	return instance.SetProperty("ReplicationCount", (value))
}

// GetReplicationCount gets the value of ReplicationCount for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyReplicationCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReplicationCount")
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

// SetReplicationLatency sets the value of ReplicationLatency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyReplicationLatency(value uint64) (err error) {
	return instance.SetProperty("ReplicationLatency", (value))
}

// GetReplicationLatency gets the value of ReplicationLatency for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyReplicationLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReplicationLatency")
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

// SetResynchronizedBytes sets the value of ResynchronizedBytes for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) SetPropertyResynchronizedBytes(value uint64) (err error) {
	return instance.SetProperty("ResynchronizedBytes", (value))
}

// GetResynchronizedBytes gets the value of ResynchronizedBytes for the instance
func (instance *Win32_PerfRawData_HyperVReplicaStats_HyperVReplicaVM) GetPropertyResynchronizedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResynchronizedBytes")
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
