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

// Win32_PerfFormattedData_Tcpip_ICMP struct
type Win32_PerfFormattedData_Tcpip_ICMP struct {
	*Win32_PerfFormattedData

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
	ReceivedAddressMask uint32

	//
	ReceivedAddressMaskReply uint32

	//
	ReceivedDestUnreachable uint32

	//
	ReceivedEchoPersec uint32

	//
	ReceivedEchoReplyPersec uint32

	//
	ReceivedParameterProblem uint32

	//
	ReceivedRedirectPersec uint32

	//
	ReceivedSourceQuench uint32

	//
	ReceivedTimeExceeded uint32

	//
	ReceivedTimestampPersec uint32

	//
	ReceivedTimestampReplyPersec uint32

	//
	SentAddressMask uint32

	//
	SentAddressMaskReply uint32

	//
	SentDestinationUnreachable uint32

	//
	SentEchoPersec uint32

	//
	SentEchoReplyPersec uint32

	//
	SentParameterProblem uint32

	//
	SentRedirectPersec uint32

	//
	SentSourceQuench uint32

	//
	SentTimeExceeded uint32

	//
	SentTimestampPersec uint32

	//
	SentTimestampReplyPersec uint32
}

func NewWin32_PerfFormattedData_Tcpip_ICMPEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Tcpip_ICMP, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_ICMP{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Tcpip_ICMPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Tcpip_ICMP, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_ICMP{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetMessagesOutboundErrors sets the value of MessagesOutboundErrors for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyMessagesOutboundErrors(value uint32) (err error) {
	return instance.SetProperty("MessagesOutboundErrors", (value))
}

// GetMessagesOutboundErrors gets the value of MessagesOutboundErrors for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyMessagesOutboundErrors() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyMessagesPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesPersec", (value))
}

// GetMessagesPersec gets the value of MessagesPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyMessagesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyMessagesReceivedErrors(value uint32) (err error) {
	return instance.SetProperty("MessagesReceivedErrors", (value))
}

// GetMessagesReceivedErrors gets the value of MessagesReceivedErrors for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyMessagesReceivedErrors() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyMessagesReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesReceivedPersec", (value))
}

// GetMessagesReceivedPersec gets the value of MessagesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyMessagesReceivedPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyMessagesSentPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesSentPersec", (value))
}

// GetMessagesSentPersec gets the value of MessagesSentPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyMessagesSentPersec() (value uint32, err error) {
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

// SetReceivedAddressMask sets the value of ReceivedAddressMask for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedAddressMask(value uint32) (err error) {
	return instance.SetProperty("ReceivedAddressMask", (value))
}

// GetReceivedAddressMask gets the value of ReceivedAddressMask for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedAddressMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedAddressMask")
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

// SetReceivedAddressMaskReply sets the value of ReceivedAddressMaskReply for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedAddressMaskReply(value uint32) (err error) {
	return instance.SetProperty("ReceivedAddressMaskReply", (value))
}

// GetReceivedAddressMaskReply gets the value of ReceivedAddressMaskReply for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedAddressMaskReply() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedAddressMaskReply")
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedDestUnreachable(value uint32) (err error) {
	return instance.SetProperty("ReceivedDestUnreachable", (value))
}

// GetReceivedDestUnreachable gets the value of ReceivedDestUnreachable for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedDestUnreachable() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedEchoPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedEchoPersec", (value))
}

// GetReceivedEchoPersec gets the value of ReceivedEchoPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedEchoPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedEchoReplyPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedEchoReplyPersec", (value))
}

// GetReceivedEchoReplyPersec gets the value of ReceivedEchoReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedEchoReplyPersec() (value uint32, err error) {
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

// SetReceivedParameterProblem sets the value of ReceivedParameterProblem for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedParameterProblem(value uint32) (err error) {
	return instance.SetProperty("ReceivedParameterProblem", (value))
}

// GetReceivedParameterProblem gets the value of ReceivedParameterProblem for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedParameterProblem() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedRedirectPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedRedirectPersec", (value))
}

// GetReceivedRedirectPersec gets the value of ReceivedRedirectPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedRedirectPersec() (value uint32, err error) {
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

// SetReceivedSourceQuench sets the value of ReceivedSourceQuench for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedSourceQuench(value uint32) (err error) {
	return instance.SetProperty("ReceivedSourceQuench", (value))
}

// GetReceivedSourceQuench gets the value of ReceivedSourceQuench for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedSourceQuench() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedSourceQuench")
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedTimeExceeded(value uint32) (err error) {
	return instance.SetProperty("ReceivedTimeExceeded", (value))
}

// GetReceivedTimeExceeded gets the value of ReceivedTimeExceeded for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedTimeExceeded() (value uint32, err error) {
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

// SetReceivedTimestampPersec sets the value of ReceivedTimestampPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedTimestampPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedTimestampPersec", (value))
}

// GetReceivedTimestampPersec gets the value of ReceivedTimestampPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedTimestampPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedTimestampPersec")
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

// SetReceivedTimestampReplyPersec sets the value of ReceivedTimestampReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertyReceivedTimestampReplyPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivedTimestampReplyPersec", (value))
}

// GetReceivedTimestampReplyPersec gets the value of ReceivedTimestampReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertyReceivedTimestampReplyPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivedTimestampReplyPersec")
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

// SetSentAddressMask sets the value of SentAddressMask for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentAddressMask(value uint32) (err error) {
	return instance.SetProperty("SentAddressMask", (value))
}

// GetSentAddressMask gets the value of SentAddressMask for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentAddressMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentAddressMask")
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

// SetSentAddressMaskReply sets the value of SentAddressMaskReply for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentAddressMaskReply(value uint32) (err error) {
	return instance.SetProperty("SentAddressMaskReply", (value))
}

// GetSentAddressMaskReply gets the value of SentAddressMaskReply for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentAddressMaskReply() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentAddressMaskReply")
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentDestinationUnreachable(value uint32) (err error) {
	return instance.SetProperty("SentDestinationUnreachable", (value))
}

// GetSentDestinationUnreachable gets the value of SentDestinationUnreachable for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentDestinationUnreachable() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentEchoPersec(value uint32) (err error) {
	return instance.SetProperty("SentEchoPersec", (value))
}

// GetSentEchoPersec gets the value of SentEchoPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentEchoPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentEchoReplyPersec(value uint32) (err error) {
	return instance.SetProperty("SentEchoReplyPersec", (value))
}

// GetSentEchoReplyPersec gets the value of SentEchoReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentEchoReplyPersec() (value uint32, err error) {
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

// SetSentParameterProblem sets the value of SentParameterProblem for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentParameterProblem(value uint32) (err error) {
	return instance.SetProperty("SentParameterProblem", (value))
}

// GetSentParameterProblem gets the value of SentParameterProblem for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentParameterProblem() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentRedirectPersec(value uint32) (err error) {
	return instance.SetProperty("SentRedirectPersec", (value))
}

// GetSentRedirectPersec gets the value of SentRedirectPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentRedirectPersec() (value uint32, err error) {
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

// SetSentSourceQuench sets the value of SentSourceQuench for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentSourceQuench(value uint32) (err error) {
	return instance.SetProperty("SentSourceQuench", (value))
}

// GetSentSourceQuench gets the value of SentSourceQuench for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentSourceQuench() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentSourceQuench")
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
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentTimeExceeded(value uint32) (err error) {
	return instance.SetProperty("SentTimeExceeded", (value))
}

// GetSentTimeExceeded gets the value of SentTimeExceeded for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentTimeExceeded() (value uint32, err error) {
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

// SetSentTimestampPersec sets the value of SentTimestampPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentTimestampPersec(value uint32) (err error) {
	return instance.SetProperty("SentTimestampPersec", (value))
}

// GetSentTimestampPersec gets the value of SentTimestampPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentTimestampPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentTimestampPersec")
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

// SetSentTimestampReplyPersec sets the value of SentTimestampReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) SetPropertySentTimestampReplyPersec(value uint32) (err error) {
	return instance.SetProperty("SentTimestampReplyPersec", (value))
}

// GetSentTimestampReplyPersec gets the value of SentTimestampReplyPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_ICMP) GetPropertySentTimestampReplyPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentTimestampReplyPersec")
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
