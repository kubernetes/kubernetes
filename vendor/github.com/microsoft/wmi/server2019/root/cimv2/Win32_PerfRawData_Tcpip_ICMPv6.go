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

// Win32_PerfRawData_Tcpip_ICMPv6 struct
type Win32_PerfRawData_Tcpip_ICMPv6 struct {
	*Win32_PerfRawData

	//
	MessagesOutboundErrors uint32

	//
	MessagesPersec uint32

	//
	MessagesReceivedErrors uint32

	//
	MessagesReceivedPersec uint32

	//
	MessagesSentPersec uint32

	//
	ReceivedDestUnreachable uint32

	//
	ReceivedEchoPersec uint32

	//
	ReceivedEchoReplyPersec uint32

	//
	ReceivedMembershipQuery uint32

	//
	ReceivedMembershipReduction uint32

	//
	ReceivedMembershipReport uint32

	//
	ReceivedNeighborAdvert uint32

	//
	ReceivedNeighborSolicit uint32

	//
	ReceivedPacketTooBig uint32

	//
	ReceivedParameterProblem uint32

	//
	ReceivedRedirectPersec uint32

	//
	ReceivedRouterAdvert uint32

	//
	ReceivedRouterSolicit uint32

	//
	ReceivedTimeExceeded uint32

	//
	SentDestinationUnreachable uint32

	//
	SentEchoPersec uint32

	//
	SentEchoReplyPersec uint32

	//
	SentMembershipQuery uint32

	//
	SentMembershipReduction uint32

	//
	SentMembershipReport uint32

	//
	SentNeighborAdvert uint32

	//
	SentNeighborSolicit uint32

	//
	SentPacketTooBig uint32

	//
	SentParameterProblem uint32

	//
	SentRedirectPersec uint32

	//
	SentRouterAdvert uint32

	//
	SentRouterSolicit uint32

	//
	SentTimeExceeded uint32
}

func NewWin32_PerfRawData_Tcpip_ICMPv6Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Tcpip_ICMPv6, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_ICMPv6{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Tcpip_ICMPv6Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Tcpip_ICMPv6, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_ICMPv6{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetMessagesOutboundErrors sets the value of MessagesOutboundErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyMessagesOutboundErrors(value uint32) (err error) {
	return instance.SetProperty("MessagesOutboundErrors", (value))
}

// GetMessagesOutboundErrors gets the value of MessagesOutboundErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyMessagesOutboundErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesOutboundErrors")
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

// SetMessagesPersec sets the value of MessagesPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyMessagesPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesPersec", (value))
}

// GetMessagesPersec gets the value of MessagesPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyMessagesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesPersec")
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

// SetMessagesReceivedErrors sets the value of MessagesReceivedErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyMessagesReceivedErrors(value uint32) (err error) {
	return instance.SetProperty("MessagesReceivedErrors", (value))
}

// GetMessagesReceivedErrors gets the value of MessagesReceivedErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyMessagesReceivedErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesReceivedErrors")
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

// SetMessagesReceivedPersec sets the value of MessagesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyMessagesReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesReceivedPersec", (value))
}

// GetMessagesReceivedPersec gets the value of MessagesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyMessagesReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesReceivedPersec")
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

// SetMessagesSentPersec sets the value of MessagesSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyMessagesSentPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesSentPersec", (value))
}

// GetMessagesSentPersec gets the value of MessagesSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyMessagesSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesSentPersec")
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

// SetReceivedDestUnreachable sets the value of ReceivedDestUnreachable for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedDestUnreachable(value uint32) (err error) {
	return instance.SetProperty("ReceivedDestUnreachable", (value))
}

// GetReceivedDestUnreachable gets the value of ReceivedDestUnreachable for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedDestUnreachable() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedDestUnreachable")
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

// SetReceivedEchoPersec sets the value of ReceivedEchoPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedEchoPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedEchoPersec", (value))
}

// GetReceivedEchoPersec gets the value of ReceivedEchoPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedEchoPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedEchoPersec")
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

// SetReceivedEchoReplyPersec sets the value of ReceivedEchoReplyPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedEchoReplyPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedEchoReplyPersec", (value))
}

// GetReceivedEchoReplyPersec gets the value of ReceivedEchoReplyPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedEchoReplyPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedEchoReplyPersec")
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

// SetReceivedMembershipQuery sets the value of ReceivedMembershipQuery for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedMembershipQuery(value uint32) (err error) {
	return instance.SetProperty("ReceivedMembershipQuery", (value))
}

// GetReceivedMembershipQuery gets the value of ReceivedMembershipQuery for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedMembershipQuery() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedMembershipQuery")
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

// SetReceivedMembershipReduction sets the value of ReceivedMembershipReduction for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedMembershipReduction(value uint32) (err error) {
	return instance.SetProperty("ReceivedMembershipReduction", (value))
}

// GetReceivedMembershipReduction gets the value of ReceivedMembershipReduction for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedMembershipReduction() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedMembershipReduction")
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

// SetReceivedMembershipReport sets the value of ReceivedMembershipReport for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedMembershipReport(value uint32) (err error) {
	return instance.SetProperty("ReceivedMembershipReport", (value))
}

// GetReceivedMembershipReport gets the value of ReceivedMembershipReport for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedMembershipReport() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedMembershipReport")
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

// SetReceivedNeighborAdvert sets the value of ReceivedNeighborAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedNeighborAdvert(value uint32) (err error) {
	return instance.SetProperty("ReceivedNeighborAdvert", (value))
}

// GetReceivedNeighborAdvert gets the value of ReceivedNeighborAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedNeighborAdvert() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedNeighborAdvert")
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

// SetReceivedNeighborSolicit sets the value of ReceivedNeighborSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedNeighborSolicit(value uint32) (err error) {
	return instance.SetProperty("ReceivedNeighborSolicit", (value))
}

// GetReceivedNeighborSolicit gets the value of ReceivedNeighborSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedNeighborSolicit() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedNeighborSolicit")
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

// SetReceivedPacketTooBig sets the value of ReceivedPacketTooBig for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedPacketTooBig(value uint32) (err error) {
	return instance.SetProperty("ReceivedPacketTooBig", (value))
}

// GetReceivedPacketTooBig gets the value of ReceivedPacketTooBig for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedPacketTooBig() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedPacketTooBig")
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

// SetReceivedParameterProblem sets the value of ReceivedParameterProblem for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedParameterProblem(value uint32) (err error) {
	return instance.SetProperty("ReceivedParameterProblem", (value))
}

// GetReceivedParameterProblem gets the value of ReceivedParameterProblem for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedParameterProblem() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedParameterProblem")
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

// SetReceivedRedirectPersec sets the value of ReceivedRedirectPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedRedirectPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedRedirectPersec", (value))
}

// GetReceivedRedirectPersec gets the value of ReceivedRedirectPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedRedirectPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedRedirectPersec")
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

// SetReceivedRouterAdvert sets the value of ReceivedRouterAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedRouterAdvert(value uint32) (err error) {
	return instance.SetProperty("ReceivedRouterAdvert", (value))
}

// GetReceivedRouterAdvert gets the value of ReceivedRouterAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedRouterAdvert() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedRouterAdvert")
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

// SetReceivedRouterSolicit sets the value of ReceivedRouterSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedRouterSolicit(value uint32) (err error) {
	return instance.SetProperty("ReceivedRouterSolicit", (value))
}

// GetReceivedRouterSolicit gets the value of ReceivedRouterSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedRouterSolicit() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedRouterSolicit")
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

// SetReceivedTimeExceeded sets the value of ReceivedTimeExceeded for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertyReceivedTimeExceeded(value uint32) (err error) {
	return instance.SetProperty("ReceivedTimeExceeded", (value))
}

// GetReceivedTimeExceeded gets the value of ReceivedTimeExceeded for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertyReceivedTimeExceeded() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedTimeExceeded")
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

// SetSentDestinationUnreachable sets the value of SentDestinationUnreachable for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentDestinationUnreachable(value uint32) (err error) {
	return instance.SetProperty("SentDestinationUnreachable", (value))
}

// GetSentDestinationUnreachable gets the value of SentDestinationUnreachable for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentDestinationUnreachable() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentDestinationUnreachable")
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

// SetSentEchoPersec sets the value of SentEchoPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentEchoPersec(value uint32) (err error) {
	return instance.SetProperty("SentEchoPersec", (value))
}

// GetSentEchoPersec gets the value of SentEchoPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentEchoPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentEchoPersec")
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

// SetSentEchoReplyPersec sets the value of SentEchoReplyPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentEchoReplyPersec(value uint32) (err error) {
	return instance.SetProperty("SentEchoReplyPersec", (value))
}

// GetSentEchoReplyPersec gets the value of SentEchoReplyPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentEchoReplyPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentEchoReplyPersec")
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

// SetSentMembershipQuery sets the value of SentMembershipQuery for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentMembershipQuery(value uint32) (err error) {
	return instance.SetProperty("SentMembershipQuery", (value))
}

// GetSentMembershipQuery gets the value of SentMembershipQuery for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentMembershipQuery() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentMembershipQuery")
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

// SetSentMembershipReduction sets the value of SentMembershipReduction for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentMembershipReduction(value uint32) (err error) {
	return instance.SetProperty("SentMembershipReduction", (value))
}

// GetSentMembershipReduction gets the value of SentMembershipReduction for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentMembershipReduction() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentMembershipReduction")
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

// SetSentMembershipReport sets the value of SentMembershipReport for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentMembershipReport(value uint32) (err error) {
	return instance.SetProperty("SentMembershipReport", (value))
}

// GetSentMembershipReport gets the value of SentMembershipReport for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentMembershipReport() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentMembershipReport")
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

// SetSentNeighborAdvert sets the value of SentNeighborAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentNeighborAdvert(value uint32) (err error) {
	return instance.SetProperty("SentNeighborAdvert", (value))
}

// GetSentNeighborAdvert gets the value of SentNeighborAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentNeighborAdvert() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentNeighborAdvert")
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

// SetSentNeighborSolicit sets the value of SentNeighborSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentNeighborSolicit(value uint32) (err error) {
	return instance.SetProperty("SentNeighborSolicit", (value))
}

// GetSentNeighborSolicit gets the value of SentNeighborSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentNeighborSolicit() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentNeighborSolicit")
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

// SetSentPacketTooBig sets the value of SentPacketTooBig for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentPacketTooBig(value uint32) (err error) {
	return instance.SetProperty("SentPacketTooBig", (value))
}

// GetSentPacketTooBig gets the value of SentPacketTooBig for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentPacketTooBig() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentPacketTooBig")
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

// SetSentParameterProblem sets the value of SentParameterProblem for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentParameterProblem(value uint32) (err error) {
	return instance.SetProperty("SentParameterProblem", (value))
}

// GetSentParameterProblem gets the value of SentParameterProblem for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentParameterProblem() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentParameterProblem")
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

// SetSentRedirectPersec sets the value of SentRedirectPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentRedirectPersec(value uint32) (err error) {
	return instance.SetProperty("SentRedirectPersec", (value))
}

// GetSentRedirectPersec gets the value of SentRedirectPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentRedirectPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRedirectPersec")
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

// SetSentRouterAdvert sets the value of SentRouterAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentRouterAdvert(value uint32) (err error) {
	return instance.SetProperty("SentRouterAdvert", (value))
}

// GetSentRouterAdvert gets the value of SentRouterAdvert for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentRouterAdvert() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRouterAdvert")
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

// SetSentRouterSolicit sets the value of SentRouterSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentRouterSolicit(value uint32) (err error) {
	return instance.SetProperty("SentRouterSolicit", (value))
}

// GetSentRouterSolicit gets the value of SentRouterSolicit for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentRouterSolicit() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRouterSolicit")
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

// SetSentTimeExceeded sets the value of SentTimeExceeded for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) SetPropertySentTimeExceeded(value uint32) (err error) {
	return instance.SetProperty("SentTimeExceeded", (value))
}

// GetSentTimeExceeded gets the value of SentTimeExceeded for the instance
func (instance *Win32_PerfRawData_Tcpip_ICMPv6) GetPropertySentTimeExceeded() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentTimeExceeded")
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
