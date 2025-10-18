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

// Win32_PerfRawData_WnvCounters_NetworkVirtualization struct
type Win32_PerfRawData_WnvCounters_NetworkVirtualization struct {
	*Win32_PerfRawData

	//
	Broadcastpacketsreceived uint64

	//
	Broadcastpacketssent uint64

	//
	InboundPacketsdropped uint64

	//
	Missingpolicyicmperrorsreceived uint64

	//
	Missingpolicyicmperrorssent uint64

	//
	Missingpolicynotificationsdropped uint64

	//
	Missingpolicynotificationsindicated uint64

	//
	Multicastpacketsreceived uint64

	//
	Multicastpacketssent uint64

	//
	OutboundPacketsdropped uint64

	//
	Packetsbuffered uint64

	//
	Packetsforwarded uint64

	//
	Packetsloopedback uint64

	//
	Policycachehits uint64

	//
	Policycachemisses uint64

	//
	Policylookupfailures uint64

	//
	Provideraddressduplicatedetectionfailures uint64

	//
	UnicastpacketsreceivedGRE uint64

	//
	UnicastpacketssentGRE uint64

	//
	UnicastReplicatedPacketsout uint64
}

func NewWin32_PerfRawData_WnvCounters_NetworkVirtualizationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_WnvCounters_NetworkVirtualization, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WnvCounters_NetworkVirtualization{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_WnvCounters_NetworkVirtualizationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_WnvCounters_NetworkVirtualization, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WnvCounters_NetworkVirtualization{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBroadcastpacketsreceived sets the value of Broadcastpacketsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyBroadcastpacketsreceived(value uint64) (err error) {
	return instance.SetProperty("Broadcastpacketsreceived", (value))
}

// GetBroadcastpacketsreceived gets the value of Broadcastpacketsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyBroadcastpacketsreceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("Broadcastpacketsreceived")
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

// SetBroadcastpacketssent sets the value of Broadcastpacketssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyBroadcastpacketssent(value uint64) (err error) {
	return instance.SetProperty("Broadcastpacketssent", (value))
}

// GetBroadcastpacketssent gets the value of Broadcastpacketssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyBroadcastpacketssent() (value uint64, err error) {
	retValue, err := instance.GetProperty("Broadcastpacketssent")
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

// SetInboundPacketsdropped sets the value of InboundPacketsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyInboundPacketsdropped(value uint64) (err error) {
	return instance.SetProperty("InboundPacketsdropped", (value))
}

// GetInboundPacketsdropped gets the value of InboundPacketsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyInboundPacketsdropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundPacketsdropped")
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

// SetMissingpolicyicmperrorsreceived sets the value of Missingpolicyicmperrorsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMissingpolicyicmperrorsreceived(value uint64) (err error) {
	return instance.SetProperty("Missingpolicyicmperrorsreceived", (value))
}

// GetMissingpolicyicmperrorsreceived gets the value of Missingpolicyicmperrorsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMissingpolicyicmperrorsreceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("Missingpolicyicmperrorsreceived")
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

// SetMissingpolicyicmperrorssent sets the value of Missingpolicyicmperrorssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMissingpolicyicmperrorssent(value uint64) (err error) {
	return instance.SetProperty("Missingpolicyicmperrorssent", (value))
}

// GetMissingpolicyicmperrorssent gets the value of Missingpolicyicmperrorssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMissingpolicyicmperrorssent() (value uint64, err error) {
	retValue, err := instance.GetProperty("Missingpolicyicmperrorssent")
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

// SetMissingpolicynotificationsdropped sets the value of Missingpolicynotificationsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMissingpolicynotificationsdropped(value uint64) (err error) {
	return instance.SetProperty("Missingpolicynotificationsdropped", (value))
}

// GetMissingpolicynotificationsdropped gets the value of Missingpolicynotificationsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMissingpolicynotificationsdropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("Missingpolicynotificationsdropped")
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

// SetMissingpolicynotificationsindicated sets the value of Missingpolicynotificationsindicated for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMissingpolicynotificationsindicated(value uint64) (err error) {
	return instance.SetProperty("Missingpolicynotificationsindicated", (value))
}

// GetMissingpolicynotificationsindicated gets the value of Missingpolicynotificationsindicated for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMissingpolicynotificationsindicated() (value uint64, err error) {
	retValue, err := instance.GetProperty("Missingpolicynotificationsindicated")
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

// SetMulticastpacketsreceived sets the value of Multicastpacketsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMulticastpacketsreceived(value uint64) (err error) {
	return instance.SetProperty("Multicastpacketsreceived", (value))
}

// GetMulticastpacketsreceived gets the value of Multicastpacketsreceived for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMulticastpacketsreceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("Multicastpacketsreceived")
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

// SetMulticastpacketssent sets the value of Multicastpacketssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyMulticastpacketssent(value uint64) (err error) {
	return instance.SetProperty("Multicastpacketssent", (value))
}

// GetMulticastpacketssent gets the value of Multicastpacketssent for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyMulticastpacketssent() (value uint64, err error) {
	retValue, err := instance.GetProperty("Multicastpacketssent")
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

// SetOutboundPacketsdropped sets the value of OutboundPacketsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyOutboundPacketsdropped(value uint64) (err error) {
	return instance.SetProperty("OutboundPacketsdropped", (value))
}

// GetOutboundPacketsdropped gets the value of OutboundPacketsdropped for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyOutboundPacketsdropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutboundPacketsdropped")
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

// SetPacketsbuffered sets the value of Packetsbuffered for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPacketsbuffered(value uint64) (err error) {
	return instance.SetProperty("Packetsbuffered", (value))
}

// GetPacketsbuffered gets the value of Packetsbuffered for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPacketsbuffered() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetsbuffered")
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

// SetPacketsforwarded sets the value of Packetsforwarded for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPacketsforwarded(value uint64) (err error) {
	return instance.SetProperty("Packetsforwarded", (value))
}

// GetPacketsforwarded gets the value of Packetsforwarded for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPacketsforwarded() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetsforwarded")
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

// SetPacketsloopedback sets the value of Packetsloopedback for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPacketsloopedback(value uint64) (err error) {
	return instance.SetProperty("Packetsloopedback", (value))
}

// GetPacketsloopedback gets the value of Packetsloopedback for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPacketsloopedback() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetsloopedback")
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

// SetPolicycachehits sets the value of Policycachehits for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPolicycachehits(value uint64) (err error) {
	return instance.SetProperty("Policycachehits", (value))
}

// GetPolicycachehits gets the value of Policycachehits for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPolicycachehits() (value uint64, err error) {
	retValue, err := instance.GetProperty("Policycachehits")
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

// SetPolicycachemisses sets the value of Policycachemisses for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPolicycachemisses(value uint64) (err error) {
	return instance.SetProperty("Policycachemisses", (value))
}

// GetPolicycachemisses gets the value of Policycachemisses for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPolicycachemisses() (value uint64, err error) {
	retValue, err := instance.GetProperty("Policycachemisses")
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

// SetPolicylookupfailures sets the value of Policylookupfailures for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyPolicylookupfailures(value uint64) (err error) {
	return instance.SetProperty("Policylookupfailures", (value))
}

// GetPolicylookupfailures gets the value of Policylookupfailures for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyPolicylookupfailures() (value uint64, err error) {
	retValue, err := instance.GetProperty("Policylookupfailures")
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

// SetProvideraddressduplicatedetectionfailures sets the value of Provideraddressduplicatedetectionfailures for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyProvideraddressduplicatedetectionfailures(value uint64) (err error) {
	return instance.SetProperty("Provideraddressduplicatedetectionfailures", (value))
}

// GetProvideraddressduplicatedetectionfailures gets the value of Provideraddressduplicatedetectionfailures for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyProvideraddressduplicatedetectionfailures() (value uint64, err error) {
	retValue, err := instance.GetProperty("Provideraddressduplicatedetectionfailures")
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

// SetUnicastpacketsreceivedGRE sets the value of UnicastpacketsreceivedGRE for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyUnicastpacketsreceivedGRE(value uint64) (err error) {
	return instance.SetProperty("UnicastpacketsreceivedGRE", (value))
}

// GetUnicastpacketsreceivedGRE gets the value of UnicastpacketsreceivedGRE for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyUnicastpacketsreceivedGRE() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnicastpacketsreceivedGRE")
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

// SetUnicastpacketssentGRE sets the value of UnicastpacketssentGRE for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyUnicastpacketssentGRE(value uint64) (err error) {
	return instance.SetProperty("UnicastpacketssentGRE", (value))
}

// GetUnicastpacketssentGRE gets the value of UnicastpacketssentGRE for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyUnicastpacketssentGRE() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnicastpacketssentGRE")
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

// SetUnicastReplicatedPacketsout sets the value of UnicastReplicatedPacketsout for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) SetPropertyUnicastReplicatedPacketsout(value uint64) (err error) {
	return instance.SetProperty("UnicastReplicatedPacketsout", (value))
}

// GetUnicastReplicatedPacketsout gets the value of UnicastReplicatedPacketsout for the instance
func (instance *Win32_PerfRawData_WnvCounters_NetworkVirtualization) GetPropertyUnicastReplicatedPacketsout() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnicastReplicatedPacketsout")
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
