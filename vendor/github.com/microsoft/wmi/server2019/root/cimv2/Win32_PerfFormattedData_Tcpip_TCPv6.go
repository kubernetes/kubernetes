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

// Win32_PerfFormattedData_Tcpip_TCPv6 struct
type Win32_PerfFormattedData_Tcpip_TCPv6 struct {
	*Win32_PerfFormattedData

	//
	ConnectionFailures uint32

	//
	ConnectionsActive uint32

	//
	ConnectionsEstablished uint32

	//
	ConnectionsPassive uint32

	//
	ConnectionsReset uint32

	//
	SegmentsPersec uint32

	//
	SegmentsReceivedPersec uint32

	//
	SegmentsRetransmittedPersec uint32

	//
	SegmentsSentPersec uint32
}

func NewWin32_PerfFormattedData_Tcpip_TCPv6Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Tcpip_TCPv6, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_TCPv6{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Tcpip_TCPv6Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Tcpip_TCPv6, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_TCPv6{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetConnectionFailures sets the value of ConnectionFailures for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertyConnectionFailures(value uint32) (err error) {
	return instance.SetProperty("ConnectionFailures", (value))
}

// GetConnectionFailures gets the value of ConnectionFailures for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertyConnectionFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionFailures")
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

// SetConnectionsActive sets the value of ConnectionsActive for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertyConnectionsActive(value uint32) (err error) {
	return instance.SetProperty("ConnectionsActive", (value))
}

// GetConnectionsActive gets the value of ConnectionsActive for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertyConnectionsActive() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsActive")
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

// SetConnectionsEstablished sets the value of ConnectionsEstablished for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertyConnectionsEstablished(value uint32) (err error) {
	return instance.SetProperty("ConnectionsEstablished", (value))
}

// GetConnectionsEstablished gets the value of ConnectionsEstablished for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertyConnectionsEstablished() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsEstablished")
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

// SetConnectionsPassive sets the value of ConnectionsPassive for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertyConnectionsPassive(value uint32) (err error) {
	return instance.SetProperty("ConnectionsPassive", (value))
}

// GetConnectionsPassive gets the value of ConnectionsPassive for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertyConnectionsPassive() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsPassive")
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

// SetConnectionsReset sets the value of ConnectionsReset for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertyConnectionsReset(value uint32) (err error) {
	return instance.SetProperty("ConnectionsReset", (value))
}

// GetConnectionsReset gets the value of ConnectionsReset for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertyConnectionsReset() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsReset")
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

// SetSegmentsPersec sets the value of SegmentsPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertySegmentsPersec(value uint32) (err error) {
	return instance.SetProperty("SegmentsPersec", (value))
}

// GetSegmentsPersec gets the value of SegmentsPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertySegmentsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SegmentsPersec")
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

// SetSegmentsReceivedPersec sets the value of SegmentsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertySegmentsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("SegmentsReceivedPersec", (value))
}

// GetSegmentsReceivedPersec gets the value of SegmentsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertySegmentsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SegmentsReceivedPersec")
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

// SetSegmentsRetransmittedPersec sets the value of SegmentsRetransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertySegmentsRetransmittedPersec(value uint32) (err error) {
	return instance.SetProperty("SegmentsRetransmittedPersec", (value))
}

// GetSegmentsRetransmittedPersec gets the value of SegmentsRetransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertySegmentsRetransmittedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SegmentsRetransmittedPersec")
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

// SetSegmentsSentPersec sets the value of SegmentsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) SetPropertySegmentsSentPersec(value uint32) (err error) {
	return instance.SetProperty("SegmentsSentPersec", (value))
}

// GetSegmentsSentPersec gets the value of SegmentsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_TCPv6) GetPropertySegmentsSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SegmentsSentPersec")
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
