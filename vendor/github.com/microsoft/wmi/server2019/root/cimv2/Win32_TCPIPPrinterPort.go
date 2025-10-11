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

// Win32_TCPIPPrinterPort struct
type Win32_TCPIPPrinterPort struct {
	*CIM_ServiceAccessPoint

	// The ByteCount property, when true, causes the computer to count the number of bytes in a document before sending them to the printer and the printer to report back the number of bytes actually read.  This is used for diagnostics when one discovers that bytes are missing from the print output.
	ByteCount bool

	// The HostAddress property indicates the address of device or print server
	HostAddress string

	// The PortNumber property indicates the number of the TCP port used by the port monitor to communitcate with the device.
	PortNumber uint32

	// The Protocol property has two values: 'Raw' indicates printing directly to a device and 'Lpr' indicates printing to device or print server; LPR is a legacy protocol, which will eventually be replaced by RAW. Some printers support only LPR.
	Protocol TCPIPPrinterPort_Protocol

	// The Queue property is used with the LPR protocol to indicate the name of the print queue on the server.
	Queue string

	// The SNMPCommunity property contains a security level value for the device.  For example 'public'.
	SNMPCommunity string

	// The property SNMPDevIndex indicates the SNMP index number of this device for the SNMP agent.
	SNMPDevIndex uint32

	// The SNMPEnabled property, when true, indicates that this printer supports RFC1759 (Simple Network Management Protocol) and can provide rich status information from the device.
	SNMPEnabled bool
}

func NewWin32_TCPIPPrinterPortEx1(instance *cim.WmiInstance) (newInstance *Win32_TCPIPPrinterPort, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_TCPIPPrinterPort{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}

func NewWin32_TCPIPPrinterPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TCPIPPrinterPort, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TCPIPPrinterPort{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}

// SetByteCount sets the value of ByteCount for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertyByteCount(value bool) (err error) {
	return instance.SetProperty("ByteCount", (value))
}

// GetByteCount gets the value of ByteCount for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertyByteCount() (value bool, err error) {
	retValue, err := instance.GetProperty("ByteCount")
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

// SetHostAddress sets the value of HostAddress for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertyHostAddress(value string) (err error) {
	return instance.SetProperty("HostAddress", (value))
}

// GetHostAddress gets the value of HostAddress for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertyHostAddress() (value string, err error) {
	retValue, err := instance.GetProperty("HostAddress")
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

// SetPortNumber sets the value of PortNumber for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertyPortNumber(value uint32) (err error) {
	return instance.SetProperty("PortNumber", (value))
}

// GetPortNumber gets the value of PortNumber for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertyPortNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("PortNumber")
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

// SetProtocol sets the value of Protocol for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertyProtocol(value TCPIPPrinterPort_Protocol) (err error) {
	return instance.SetProperty("Protocol", (value))
}

// GetProtocol gets the value of Protocol for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertyProtocol() (value TCPIPPrinterPort_Protocol, err error) {
	retValue, err := instance.GetProperty("Protocol")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = TCPIPPrinterPort_Protocol(valuetmp)

	return
}

// SetQueue sets the value of Queue for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertyQueue(value string) (err error) {
	return instance.SetProperty("Queue", (value))
}

// GetQueue gets the value of Queue for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertyQueue() (value string, err error) {
	retValue, err := instance.GetProperty("Queue")
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

// SetSNMPCommunity sets the value of SNMPCommunity for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertySNMPCommunity(value string) (err error) {
	return instance.SetProperty("SNMPCommunity", (value))
}

// GetSNMPCommunity gets the value of SNMPCommunity for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertySNMPCommunity() (value string, err error) {
	retValue, err := instance.GetProperty("SNMPCommunity")
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

// SetSNMPDevIndex sets the value of SNMPDevIndex for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertySNMPDevIndex(value uint32) (err error) {
	return instance.SetProperty("SNMPDevIndex", (value))
}

// GetSNMPDevIndex gets the value of SNMPDevIndex for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertySNMPDevIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("SNMPDevIndex")
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

// SetSNMPEnabled sets the value of SNMPEnabled for the instance
func (instance *Win32_TCPIPPrinterPort) SetPropertySNMPEnabled(value bool) (err error) {
	return instance.SetProperty("SNMPEnabled", (value))
}

// GetSNMPEnabled gets the value of SNMPEnabled for the instance
func (instance *Win32_TCPIPPrinterPort) GetPropertySNMPEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SNMPEnabled")
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
