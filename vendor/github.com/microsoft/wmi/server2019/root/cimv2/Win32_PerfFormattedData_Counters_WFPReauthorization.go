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

// Win32_PerfFormattedData_Counters_WFPReauthorization struct
type Win32_PerfFormattedData_Counters_WFPReauthorization struct {
	*Win32_PerfFormattedData

	//
	FamilyV4 uint64

	//
	FamilyV6 uint64

	//
	Inbound uint64

	//
	Outbound uint64

	//
	ProtocolICMP uint64

	//
	ProtocolICMP6 uint64

	//
	ProtocolIPv4 uint64

	//
	ProtocolIPv6 uint64

	//
	ProtocolOther uint64

	//
	ProtocolTCP uint64

	//
	ProtocolUDP uint64

	//
	ReasonClassifyCompletion uint64

	//
	ReasonEDPPolicyChanged uint64

	//
	ReasonIPSecPropertiesChanged uint64

	//
	ReasonMidStreamInspection uint64

	//
	ReasonNewArrivalInterface uint64

	//
	ReasonNewInboundMCastBCastPacket uint64

	//
	ReasonNewNextHopInterface uint64

	//
	ReasonPolicyChange uint64

	//
	ReasonPreclassifyLocalAddressDimensionPolicyChanged uint64

	//
	ReasonPreclassifyLocalPortDimensionPolicyChanged uint64

	//
	ReasonPreclassifyRemoteAddressDimensionPolicyChanged uint64

	//
	ReasonPreclassifyRemotePortDimensionPolicyChanged uint64

	//
	ReasonProfileCrossing uint64

	//
	ReasonProxyHandleChanged uint64

	//
	ReasonSocketPropertyChanged uint64
}

func NewWin32_PerfFormattedData_Counters_WFPReauthorizationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_WFPReauthorization, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_WFPReauthorization{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_WFPReauthorizationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_WFPReauthorization, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_WFPReauthorization{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetFamilyV4 sets the value of FamilyV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyFamilyV4(value uint64) (err error) {
	return instance.SetProperty("FamilyV4", (value))
}

// GetFamilyV4 gets the value of FamilyV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyFamilyV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FamilyV4")
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

// SetFamilyV6 sets the value of FamilyV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyFamilyV6(value uint64) (err error) {
	return instance.SetProperty("FamilyV6", (value))
}

// GetFamilyV6 gets the value of FamilyV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyFamilyV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FamilyV6")
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

// SetInbound sets the value of Inbound for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyInbound(value uint64) (err error) {
	return instance.SetProperty("Inbound", (value))
}

// GetInbound gets the value of Inbound for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyInbound() (value uint64, err error) {
	retValue, err := instance.GetProperty("Inbound")
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

// SetOutbound sets the value of Outbound for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyOutbound(value uint64) (err error) {
	return instance.SetProperty("Outbound", (value))
}

// GetOutbound gets the value of Outbound for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyOutbound() (value uint64, err error) {
	retValue, err := instance.GetProperty("Outbound")
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

// SetProtocolICMP sets the value of ProtocolICMP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolICMP(value uint64) (err error) {
	return instance.SetProperty("ProtocolICMP", (value))
}

// GetProtocolICMP gets the value of ProtocolICMP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolICMP() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolICMP")
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

// SetProtocolICMP6 sets the value of ProtocolICMP6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolICMP6(value uint64) (err error) {
	return instance.SetProperty("ProtocolICMP6", (value))
}

// GetProtocolICMP6 gets the value of ProtocolICMP6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolICMP6() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolICMP6")
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

// SetProtocolIPv4 sets the value of ProtocolIPv4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolIPv4(value uint64) (err error) {
	return instance.SetProperty("ProtocolIPv4", (value))
}

// GetProtocolIPv4 gets the value of ProtocolIPv4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolIPv4() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolIPv4")
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

// SetProtocolIPv6 sets the value of ProtocolIPv6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolIPv6(value uint64) (err error) {
	return instance.SetProperty("ProtocolIPv6", (value))
}

// GetProtocolIPv6 gets the value of ProtocolIPv6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolIPv6() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolIPv6")
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

// SetProtocolOther sets the value of ProtocolOther for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolOther(value uint64) (err error) {
	return instance.SetProperty("ProtocolOther", (value))
}

// GetProtocolOther gets the value of ProtocolOther for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolOther() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolOther")
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

// SetProtocolTCP sets the value of ProtocolTCP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolTCP(value uint64) (err error) {
	return instance.SetProperty("ProtocolTCP", (value))
}

// GetProtocolTCP gets the value of ProtocolTCP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolTCP() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolTCP")
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

// SetProtocolUDP sets the value of ProtocolUDP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyProtocolUDP(value uint64) (err error) {
	return instance.SetProperty("ProtocolUDP", (value))
}

// GetProtocolUDP gets the value of ProtocolUDP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyProtocolUDP() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProtocolUDP")
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

// SetReasonClassifyCompletion sets the value of ReasonClassifyCompletion for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonClassifyCompletion(value uint64) (err error) {
	return instance.SetProperty("ReasonClassifyCompletion", (value))
}

// GetReasonClassifyCompletion gets the value of ReasonClassifyCompletion for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonClassifyCompletion() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonClassifyCompletion")
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

// SetReasonEDPPolicyChanged sets the value of ReasonEDPPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonEDPPolicyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonEDPPolicyChanged", (value))
}

// GetReasonEDPPolicyChanged gets the value of ReasonEDPPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonEDPPolicyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonEDPPolicyChanged")
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

// SetReasonIPSecPropertiesChanged sets the value of ReasonIPSecPropertiesChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonIPSecPropertiesChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonIPSecPropertiesChanged", (value))
}

// GetReasonIPSecPropertiesChanged gets the value of ReasonIPSecPropertiesChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonIPSecPropertiesChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonIPSecPropertiesChanged")
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

// SetReasonMidStreamInspection sets the value of ReasonMidStreamInspection for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonMidStreamInspection(value uint64) (err error) {
	return instance.SetProperty("ReasonMidStreamInspection", (value))
}

// GetReasonMidStreamInspection gets the value of ReasonMidStreamInspection for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonMidStreamInspection() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonMidStreamInspection")
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

// SetReasonNewArrivalInterface sets the value of ReasonNewArrivalInterface for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonNewArrivalInterface(value uint64) (err error) {
	return instance.SetProperty("ReasonNewArrivalInterface", (value))
}

// GetReasonNewArrivalInterface gets the value of ReasonNewArrivalInterface for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonNewArrivalInterface() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonNewArrivalInterface")
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

// SetReasonNewInboundMCastBCastPacket sets the value of ReasonNewInboundMCastBCastPacket for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonNewInboundMCastBCastPacket(value uint64) (err error) {
	return instance.SetProperty("ReasonNewInboundMCastBCastPacket", (value))
}

// GetReasonNewInboundMCastBCastPacket gets the value of ReasonNewInboundMCastBCastPacket for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonNewInboundMCastBCastPacket() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonNewInboundMCastBCastPacket")
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

// SetReasonNewNextHopInterface sets the value of ReasonNewNextHopInterface for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonNewNextHopInterface(value uint64) (err error) {
	return instance.SetProperty("ReasonNewNextHopInterface", (value))
}

// GetReasonNewNextHopInterface gets the value of ReasonNewNextHopInterface for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonNewNextHopInterface() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonNewNextHopInterface")
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

// SetReasonPolicyChange sets the value of ReasonPolicyChange for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonPolicyChange(value uint64) (err error) {
	return instance.SetProperty("ReasonPolicyChange", (value))
}

// GetReasonPolicyChange gets the value of ReasonPolicyChange for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonPolicyChange() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonPolicyChange")
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

// SetReasonPreclassifyLocalAddressDimensionPolicyChanged sets the value of ReasonPreclassifyLocalAddressDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonPreclassifyLocalAddressDimensionPolicyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonPreclassifyLocalAddressDimensionPolicyChanged", (value))
}

// GetReasonPreclassifyLocalAddressDimensionPolicyChanged gets the value of ReasonPreclassifyLocalAddressDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonPreclassifyLocalAddressDimensionPolicyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonPreclassifyLocalAddressDimensionPolicyChanged")
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

// SetReasonPreclassifyLocalPortDimensionPolicyChanged sets the value of ReasonPreclassifyLocalPortDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonPreclassifyLocalPortDimensionPolicyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonPreclassifyLocalPortDimensionPolicyChanged", (value))
}

// GetReasonPreclassifyLocalPortDimensionPolicyChanged gets the value of ReasonPreclassifyLocalPortDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonPreclassifyLocalPortDimensionPolicyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonPreclassifyLocalPortDimensionPolicyChanged")
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

// SetReasonPreclassifyRemoteAddressDimensionPolicyChanged sets the value of ReasonPreclassifyRemoteAddressDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonPreclassifyRemoteAddressDimensionPolicyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonPreclassifyRemoteAddressDimensionPolicyChanged", (value))
}

// GetReasonPreclassifyRemoteAddressDimensionPolicyChanged gets the value of ReasonPreclassifyRemoteAddressDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonPreclassifyRemoteAddressDimensionPolicyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonPreclassifyRemoteAddressDimensionPolicyChanged")
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

// SetReasonPreclassifyRemotePortDimensionPolicyChanged sets the value of ReasonPreclassifyRemotePortDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonPreclassifyRemotePortDimensionPolicyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonPreclassifyRemotePortDimensionPolicyChanged", (value))
}

// GetReasonPreclassifyRemotePortDimensionPolicyChanged gets the value of ReasonPreclassifyRemotePortDimensionPolicyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonPreclassifyRemotePortDimensionPolicyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonPreclassifyRemotePortDimensionPolicyChanged")
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

// SetReasonProfileCrossing sets the value of ReasonProfileCrossing for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonProfileCrossing(value uint64) (err error) {
	return instance.SetProperty("ReasonProfileCrossing", (value))
}

// GetReasonProfileCrossing gets the value of ReasonProfileCrossing for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonProfileCrossing() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonProfileCrossing")
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

// SetReasonProxyHandleChanged sets the value of ReasonProxyHandleChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonProxyHandleChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonProxyHandleChanged", (value))
}

// GetReasonProxyHandleChanged gets the value of ReasonProxyHandleChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonProxyHandleChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonProxyHandleChanged")
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

// SetReasonSocketPropertyChanged sets the value of ReasonSocketPropertyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) SetPropertyReasonSocketPropertyChanged(value uint64) (err error) {
	return instance.SetProperty("ReasonSocketPropertyChanged", (value))
}

// GetReasonSocketPropertyChanged gets the value of ReasonSocketPropertyChanged for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPReauthorization) GetPropertyReasonSocketPropertyChanged() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReasonSocketPropertyChanged")
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
