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

// Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages struct
type Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages struct {
	*Win32_PerfRawData

	//
	BytesReceived uint64

	//
	BytesReceivedPersec uint64

	//
	BytesSent uint64

	//
	BytesSentPersec uint64

	//
	MessagesReceived uint64

	//
	MessagesReceivedPersec uint64

	//
	MessagesSent uint64

	//
	MessagesSentPersec uint64
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessagesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessagesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceived")
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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

// SetBytesSent sets the value of BytesSent for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyBytesSent(value uint64) (err error) {
	return instance.SetProperty("BytesSent", (value))
}

// GetBytesSent gets the value of BytesSent for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSent")
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
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyBytesSentPersec() (value uint64, err error) {
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

// SetMessagesReceived sets the value of MessagesReceived for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyMessagesReceived(value uint64) (err error) {
	return instance.SetProperty("MessagesReceived", (value))
}

// GetMessagesReceived gets the value of MessagesReceived for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyMessagesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("MessagesReceived")
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

// SetMessagesReceivedPersec sets the value of MessagesReceivedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyMessagesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("MessagesReceivedPersec", (value))
}

// GetMessagesReceivedPersec gets the value of MessagesReceivedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyMessagesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MessagesReceivedPersec")
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

// SetMessagesSent sets the value of MessagesSent for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyMessagesSent(value uint64) (err error) {
	return instance.SetProperty("MessagesSent", (value))
}

// GetMessagesSent gets the value of MessagesSent for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyMessagesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("MessagesSent")
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

// SetMessagesSentPersec sets the value of MessagesSentPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) SetPropertyMessagesSentPersec(value uint64) (err error) {
	return instance.SetProperty("MessagesSentPersec", (value))
}

// GetMessagesSentPersec gets the value of MessagesSentPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkMessages) GetPropertyMessagesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MessagesSentPersec")
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
