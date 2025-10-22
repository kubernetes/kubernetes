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

// Win32_PerfFormattedData_Counters_WFPClassify struct
type Win32_PerfFormattedData_Counters_WFPClassify struct {
	*Win32_PerfFormattedData

	//
	FWPMLAYERALEAUTHCONNECTV4 uint64

	//
	FWPMLAYERALEAUTHCONNECTV4DISCARD uint64

	//
	FWPMLAYERALEAUTHCONNECTV6 uint64

	//
	FWPMLAYERALEAUTHCONNECTV6DISCARD uint64

	//
	FWPMLAYERALEAUTHLISTENV4 uint64

	//
	FWPMLAYERALEAUTHLISTENV4DISCARD uint64

	//
	FWPMLAYERALEAUTHLISTENV6 uint64

	//
	FWPMLAYERALEAUTHLISTENV6DISCARD uint64

	//
	FWPMLAYERALEAUTHRECVACCEPTV4 uint64

	//
	FWPMLAYERALEAUTHRECVACCEPTV4DISCARD uint64

	//
	FWPMLAYERALEAUTHRECVACCEPTV6 uint64

	//
	FWPMLAYERALEAUTHRECVACCEPTV6DISCARD uint64

	//
	FWPMLAYERALEBINDREDIRECTV4 uint64

	//
	FWPMLAYERALEBINDREDIRECTV6 uint64

	//
	FWPMLAYERALECONNECTREDIRECTV4 uint64

	//
	FWPMLAYERALECONNECTREDIRECTV6 uint64

	//
	FWPMLAYERALEENDPOINTCLOSUREV4 uint64

	//
	FWPMLAYERALEENDPOINTCLOSUREV6 uint64

	//
	FWPMLAYERALEFLOWESTABLISHEDV4 uint64

	//
	FWPMLAYERALEFLOWESTABLISHEDV4DISCARD uint64

	//
	FWPMLAYERALEFLOWESTABLISHEDV6 uint64

	//
	FWPMLAYERALEFLOWESTABLISHEDV6DISCARD uint64

	//
	FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPLOCALPORTV4 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPLOCALPORTV6 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4 uint64

	//
	FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6 uint64

	//
	FWPMLAYERALERESOURCEASSIGNMENTV4 uint64

	//
	FWPMLAYERALERESOURCEASSIGNMENTV4DISCARD uint64

	//
	FWPMLAYERALERESOURCEASSIGNMENTV6 uint64

	//
	FWPMLAYERALERESOURCEASSIGNMENTV6DISCARD uint64

	//
	FWPMLAYERALERESOURCERELEASEV4 uint64

	//
	FWPMLAYERALERESOURCERELEASEV6 uint64

	//
	FWPMLAYERDATAGRAMDATAV4 uint64

	//
	FWPMLAYERDATAGRAMDATAV4DISCARD uint64

	//
	FWPMLAYERDATAGRAMDATAV6 uint64

	//
	FWPMLAYERDATAGRAMDATAV6DISCARD uint64

	//
	FWPMLAYEREGRESSVSWITCHETHERNET uint64

	//
	FWPMLAYEREGRESSVSWITCHTRANSPORTV4 uint64

	//
	FWPMLAYEREGRESSVSWITCHTRANSPORTV6 uint64

	//
	FWPMLAYERIKEEXTV4 uint64

	//
	FWPMLAYERIKEEXTV6 uint64

	//
	FWPMLAYERINBOUNDICMPERRORV4 uint64

	//
	FWPMLAYERINBOUNDICMPERRORV4DISCARD uint64

	//
	FWPMLAYERINBOUNDICMPERRORV6 uint64

	//
	FWPMLAYERINBOUNDICMPERRORV6DISCARD uint64

	//
	FWPMLAYERINBOUNDIPPACKETV4 uint64

	//
	FWPMLAYERINBOUNDIPPACKETV4DISCARD uint64

	//
	FWPMLAYERINBOUNDIPPACKETV6 uint64

	//
	FWPMLAYERINBOUNDIPPACKETV6DISCARD uint64

	//
	FWPMLAYERINBOUNDMACFRAMEETHERNET uint64

	//
	FWPMLAYERINBOUNDMACFRAMENATIVE uint64

	//
	FWPMLAYERINBOUNDMACFRAMENATIVEFAST uint64

	//
	FWPMLAYERINBOUNDSSLTHROTTLING uint64

	//
	FWPMLAYERINBOUNDTRANSPORTFAST uint64

	//
	FWPMLAYERINBOUNDTRANSPORTV4 uint64

	//
	FWPMLAYERINBOUNDTRANSPORTV4DISCARD uint64

	//
	FWPMLAYERINBOUNDTRANSPORTV6 uint64

	//
	FWPMLAYERINBOUNDTRANSPORTV6DISCARD uint64

	//
	FWPMLAYERINGRESSVSWITCHETHERNET uint64

	//
	FWPMLAYERINGRESSVSWITCHTRANSPORTV4 uint64

	//
	FWPMLAYERINGRESSVSWITCHTRANSPORTV6 uint64

	//
	FWPMLAYERIPFORWARDV4 uint64

	//
	FWPMLAYERIPFORWARDV4DISCARD uint64

	//
	FWPMLAYERIPFORWARDV6 uint64

	//
	FWPMLAYERIPFORWARDV6DISCARD uint64

	//
	FWPMLAYERIPSECKMDEMUXV4 uint64

	//
	FWPMLAYERIPSECKMDEMUXV6 uint64

	//
	FWPMLAYERIPSECV4 uint64

	//
	FWPMLAYERIPSECV6 uint64

	//
	FWPMLAYERKMAUTHORIZATION uint64

	//
	FWPMLAYERNAMERESOLUTIONCACHEV4 uint64

	//
	FWPMLAYERNAMERESOLUTIONCACHEV6 uint64

	//
	FWPMLAYEROUTBOUNDICMPERRORV4 uint64

	//
	FWPMLAYEROUTBOUNDICMPERRORV4DISCARD uint64

	//
	FWPMLAYEROUTBOUNDICMPERRORV6 uint64

	//
	FWPMLAYEROUTBOUNDICMPERRORV6DISCARD uint64

	//
	FWPMLAYEROUTBOUNDIPPACKETV4 uint64

	//
	FWPMLAYEROUTBOUNDIPPACKETV4DISCARD uint64

	//
	FWPMLAYEROUTBOUNDIPPACKETV6 uint64

	//
	FWPMLAYEROUTBOUNDIPPACKETV6DISCARD uint64

	//
	FWPMLAYEROUTBOUNDMACFRAMEETHERNET uint64

	//
	FWPMLAYEROUTBOUNDMACFRAMENATIVE uint64

	//
	FWPMLAYEROUTBOUNDMACFRAMENATIVEFAST uint64

	//
	FWPMLAYEROUTBOUNDTRANSPORTFAST uint64

	//
	FWPMLAYEROUTBOUNDTRANSPORTV4 uint64

	//
	FWPMLAYEROUTBOUNDTRANSPORTV4DISCARD uint64

	//
	FWPMLAYEROUTBOUNDTRANSPORTV6 uint64

	//
	FWPMLAYEROUTBOUNDTRANSPORTV6DISCARD uint64

	//
	FWPMLAYERRPCEPADD uint64

	//
	FWPMLAYERRPCEPMAP uint64

	//
	FWPMLAYERRPCPROXYCONN uint64

	//
	FWPMLAYERRPCPROXYIF uint64

	//
	FWPMLAYERRPCUM uint64

	//
	FWPMLAYERSTREAMPACKETV4 uint64

	//
	FWPMLAYERSTREAMPACKETV6 uint64

	//
	FWPMLAYERSTREAMV4 uint64

	//
	FWPMLAYERSTREAMV4DISCARD uint64

	//
	FWPMLAYERSTREAMV6 uint64

	//
	FWPMLAYERSTREAMV6DISCARD uint64

	//
	Total uint64
}

func NewWin32_PerfFormattedData_Counters_WFPClassifyEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_WFPClassify, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_WFPClassify{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_WFPClassifyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_WFPClassify, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_WFPClassify{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetFWPMLAYERALEAUTHCONNECTV4 sets the value of FWPMLAYERALEAUTHCONNECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHCONNECTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHCONNECTV4", (value))
}

// GetFWPMLAYERALEAUTHCONNECTV4 gets the value of FWPMLAYERALEAUTHCONNECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHCONNECTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHCONNECTV4")
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

// SetFWPMLAYERALEAUTHCONNECTV4DISCARD sets the value of FWPMLAYERALEAUTHCONNECTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHCONNECTV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHCONNECTV4DISCARD", (value))
}

// GetFWPMLAYERALEAUTHCONNECTV4DISCARD gets the value of FWPMLAYERALEAUTHCONNECTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHCONNECTV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHCONNECTV4DISCARD")
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

// SetFWPMLAYERALEAUTHCONNECTV6 sets the value of FWPMLAYERALEAUTHCONNECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHCONNECTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHCONNECTV6", (value))
}

// GetFWPMLAYERALEAUTHCONNECTV6 gets the value of FWPMLAYERALEAUTHCONNECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHCONNECTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHCONNECTV6")
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

// SetFWPMLAYERALEAUTHCONNECTV6DISCARD sets the value of FWPMLAYERALEAUTHCONNECTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHCONNECTV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHCONNECTV6DISCARD", (value))
}

// GetFWPMLAYERALEAUTHCONNECTV6DISCARD gets the value of FWPMLAYERALEAUTHCONNECTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHCONNECTV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHCONNECTV6DISCARD")
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

// SetFWPMLAYERALEAUTHLISTENV4 sets the value of FWPMLAYERALEAUTHLISTENV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHLISTENV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHLISTENV4", (value))
}

// GetFWPMLAYERALEAUTHLISTENV4 gets the value of FWPMLAYERALEAUTHLISTENV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHLISTENV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHLISTENV4")
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

// SetFWPMLAYERALEAUTHLISTENV4DISCARD sets the value of FWPMLAYERALEAUTHLISTENV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHLISTENV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHLISTENV4DISCARD", (value))
}

// GetFWPMLAYERALEAUTHLISTENV4DISCARD gets the value of FWPMLAYERALEAUTHLISTENV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHLISTENV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHLISTENV4DISCARD")
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

// SetFWPMLAYERALEAUTHLISTENV6 sets the value of FWPMLAYERALEAUTHLISTENV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHLISTENV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHLISTENV6", (value))
}

// GetFWPMLAYERALEAUTHLISTENV6 gets the value of FWPMLAYERALEAUTHLISTENV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHLISTENV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHLISTENV6")
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

// SetFWPMLAYERALEAUTHLISTENV6DISCARD sets the value of FWPMLAYERALEAUTHLISTENV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHLISTENV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHLISTENV6DISCARD", (value))
}

// GetFWPMLAYERALEAUTHLISTENV6DISCARD gets the value of FWPMLAYERALEAUTHLISTENV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHLISTENV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHLISTENV6DISCARD")
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

// SetFWPMLAYERALEAUTHRECVACCEPTV4 sets the value of FWPMLAYERALEAUTHRECVACCEPTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHRECVACCEPTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHRECVACCEPTV4", (value))
}

// GetFWPMLAYERALEAUTHRECVACCEPTV4 gets the value of FWPMLAYERALEAUTHRECVACCEPTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHRECVACCEPTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHRECVACCEPTV4")
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

// SetFWPMLAYERALEAUTHRECVACCEPTV4DISCARD sets the value of FWPMLAYERALEAUTHRECVACCEPTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHRECVACCEPTV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHRECVACCEPTV4DISCARD", (value))
}

// GetFWPMLAYERALEAUTHRECVACCEPTV4DISCARD gets the value of FWPMLAYERALEAUTHRECVACCEPTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHRECVACCEPTV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHRECVACCEPTV4DISCARD")
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

// SetFWPMLAYERALEAUTHRECVACCEPTV6 sets the value of FWPMLAYERALEAUTHRECVACCEPTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHRECVACCEPTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHRECVACCEPTV6", (value))
}

// GetFWPMLAYERALEAUTHRECVACCEPTV6 gets the value of FWPMLAYERALEAUTHRECVACCEPTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHRECVACCEPTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHRECVACCEPTV6")
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

// SetFWPMLAYERALEAUTHRECVACCEPTV6DISCARD sets the value of FWPMLAYERALEAUTHRECVACCEPTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEAUTHRECVACCEPTV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEAUTHRECVACCEPTV6DISCARD", (value))
}

// GetFWPMLAYERALEAUTHRECVACCEPTV6DISCARD gets the value of FWPMLAYERALEAUTHRECVACCEPTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEAUTHRECVACCEPTV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEAUTHRECVACCEPTV6DISCARD")
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

// SetFWPMLAYERALEBINDREDIRECTV4 sets the value of FWPMLAYERALEBINDREDIRECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEBINDREDIRECTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEBINDREDIRECTV4", (value))
}

// GetFWPMLAYERALEBINDREDIRECTV4 gets the value of FWPMLAYERALEBINDREDIRECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEBINDREDIRECTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEBINDREDIRECTV4")
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

// SetFWPMLAYERALEBINDREDIRECTV6 sets the value of FWPMLAYERALEBINDREDIRECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEBINDREDIRECTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEBINDREDIRECTV6", (value))
}

// GetFWPMLAYERALEBINDREDIRECTV6 gets the value of FWPMLAYERALEBINDREDIRECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEBINDREDIRECTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEBINDREDIRECTV6")
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

// SetFWPMLAYERALECONNECTREDIRECTV4 sets the value of FWPMLAYERALECONNECTREDIRECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALECONNECTREDIRECTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALECONNECTREDIRECTV4", (value))
}

// GetFWPMLAYERALECONNECTREDIRECTV4 gets the value of FWPMLAYERALECONNECTREDIRECTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALECONNECTREDIRECTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALECONNECTREDIRECTV4")
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

// SetFWPMLAYERALECONNECTREDIRECTV6 sets the value of FWPMLAYERALECONNECTREDIRECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALECONNECTREDIRECTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALECONNECTREDIRECTV6", (value))
}

// GetFWPMLAYERALECONNECTREDIRECTV6 gets the value of FWPMLAYERALECONNECTREDIRECTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALECONNECTREDIRECTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALECONNECTREDIRECTV6")
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

// SetFWPMLAYERALEENDPOINTCLOSUREV4 sets the value of FWPMLAYERALEENDPOINTCLOSUREV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEENDPOINTCLOSUREV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEENDPOINTCLOSUREV4", (value))
}

// GetFWPMLAYERALEENDPOINTCLOSUREV4 gets the value of FWPMLAYERALEENDPOINTCLOSUREV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEENDPOINTCLOSUREV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEENDPOINTCLOSUREV4")
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

// SetFWPMLAYERALEENDPOINTCLOSUREV6 sets the value of FWPMLAYERALEENDPOINTCLOSUREV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEENDPOINTCLOSUREV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEENDPOINTCLOSUREV6", (value))
}

// GetFWPMLAYERALEENDPOINTCLOSUREV6 gets the value of FWPMLAYERALEENDPOINTCLOSUREV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEENDPOINTCLOSUREV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEENDPOINTCLOSUREV6")
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

// SetFWPMLAYERALEFLOWESTABLISHEDV4 sets the value of FWPMLAYERALEFLOWESTABLISHEDV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEFLOWESTABLISHEDV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEFLOWESTABLISHEDV4", (value))
}

// GetFWPMLAYERALEFLOWESTABLISHEDV4 gets the value of FWPMLAYERALEFLOWESTABLISHEDV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEFLOWESTABLISHEDV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEFLOWESTABLISHEDV4")
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

// SetFWPMLAYERALEFLOWESTABLISHEDV4DISCARD sets the value of FWPMLAYERALEFLOWESTABLISHEDV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEFLOWESTABLISHEDV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEFLOWESTABLISHEDV4DISCARD", (value))
}

// GetFWPMLAYERALEFLOWESTABLISHEDV4DISCARD gets the value of FWPMLAYERALEFLOWESTABLISHEDV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEFLOWESTABLISHEDV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEFLOWESTABLISHEDV4DISCARD")
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

// SetFWPMLAYERALEFLOWESTABLISHEDV6 sets the value of FWPMLAYERALEFLOWESTABLISHEDV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEFLOWESTABLISHEDV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEFLOWESTABLISHEDV6", (value))
}

// GetFWPMLAYERALEFLOWESTABLISHEDV6 gets the value of FWPMLAYERALEFLOWESTABLISHEDV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEFLOWESTABLISHEDV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEFLOWESTABLISHEDV6")
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

// SetFWPMLAYERALEFLOWESTABLISHEDV6DISCARD sets the value of FWPMLAYERALEFLOWESTABLISHEDV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEFLOWESTABLISHEDV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEFLOWESTABLISHEDV6DISCARD", (value))
}

// GetFWPMLAYERALEFLOWESTABLISHEDV6DISCARD gets the value of FWPMLAYERALEFLOWESTABLISHEDV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEFLOWESTABLISHEDV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEFLOWESTABLISHEDV6DISCARD")
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

// SetFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4 sets the value of FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4 gets the value of FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV4")
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

// SetFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6 sets the value of FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6 gets the value of FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALADDRESSV6")
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

// SetFWPMLAYERALEPRECLASSIFYIPLOCALPORTV4 sets the value of FWPMLAYERALEPRECLASSIFYIPLOCALPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALPORTV4", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPLOCALPORTV4 gets the value of FWPMLAYERALEPRECLASSIFYIPLOCALPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALPORTV4")
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

// SetFWPMLAYERALEPRECLASSIFYIPLOCALPORTV6 sets the value of FWPMLAYERALEPRECLASSIFYIPLOCALPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALPORTV6", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPLOCALPORTV6 gets the value of FWPMLAYERALEPRECLASSIFYIPLOCALPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPLOCALPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPLOCALPORTV6")
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

// SetFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4 sets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4 gets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV4")
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

// SetFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6 sets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6 gets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEADDRESSV6")
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

// SetFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4 sets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4 gets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV4")
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

// SetFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6 sets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6", (value))
}

// GetFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6 gets the value of FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALEPRECLASSIFYIPREMOTEPORTV6")
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

// SetFWPMLAYERALERESOURCEASSIGNMENTV4 sets the value of FWPMLAYERALERESOURCEASSIGNMENTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCEASSIGNMENTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCEASSIGNMENTV4", (value))
}

// GetFWPMLAYERALERESOURCEASSIGNMENTV4 gets the value of FWPMLAYERALERESOURCEASSIGNMENTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCEASSIGNMENTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCEASSIGNMENTV4")
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

// SetFWPMLAYERALERESOURCEASSIGNMENTV4DISCARD sets the value of FWPMLAYERALERESOURCEASSIGNMENTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCEASSIGNMENTV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCEASSIGNMENTV4DISCARD", (value))
}

// GetFWPMLAYERALERESOURCEASSIGNMENTV4DISCARD gets the value of FWPMLAYERALERESOURCEASSIGNMENTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCEASSIGNMENTV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCEASSIGNMENTV4DISCARD")
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

// SetFWPMLAYERALERESOURCEASSIGNMENTV6 sets the value of FWPMLAYERALERESOURCEASSIGNMENTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCEASSIGNMENTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCEASSIGNMENTV6", (value))
}

// GetFWPMLAYERALERESOURCEASSIGNMENTV6 gets the value of FWPMLAYERALERESOURCEASSIGNMENTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCEASSIGNMENTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCEASSIGNMENTV6")
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

// SetFWPMLAYERALERESOURCEASSIGNMENTV6DISCARD sets the value of FWPMLAYERALERESOURCEASSIGNMENTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCEASSIGNMENTV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCEASSIGNMENTV6DISCARD", (value))
}

// GetFWPMLAYERALERESOURCEASSIGNMENTV6DISCARD gets the value of FWPMLAYERALERESOURCEASSIGNMENTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCEASSIGNMENTV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCEASSIGNMENTV6DISCARD")
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

// SetFWPMLAYERALERESOURCERELEASEV4 sets the value of FWPMLAYERALERESOURCERELEASEV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCERELEASEV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCERELEASEV4", (value))
}

// GetFWPMLAYERALERESOURCERELEASEV4 gets the value of FWPMLAYERALERESOURCERELEASEV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCERELEASEV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCERELEASEV4")
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

// SetFWPMLAYERALERESOURCERELEASEV6 sets the value of FWPMLAYERALERESOURCERELEASEV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERALERESOURCERELEASEV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERALERESOURCERELEASEV6", (value))
}

// GetFWPMLAYERALERESOURCERELEASEV6 gets the value of FWPMLAYERALERESOURCERELEASEV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERALERESOURCERELEASEV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERALERESOURCERELEASEV6")
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

// SetFWPMLAYERDATAGRAMDATAV4 sets the value of FWPMLAYERDATAGRAMDATAV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERDATAGRAMDATAV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERDATAGRAMDATAV4", (value))
}

// GetFWPMLAYERDATAGRAMDATAV4 gets the value of FWPMLAYERDATAGRAMDATAV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERDATAGRAMDATAV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERDATAGRAMDATAV4")
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

// SetFWPMLAYERDATAGRAMDATAV4DISCARD sets the value of FWPMLAYERDATAGRAMDATAV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERDATAGRAMDATAV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERDATAGRAMDATAV4DISCARD", (value))
}

// GetFWPMLAYERDATAGRAMDATAV4DISCARD gets the value of FWPMLAYERDATAGRAMDATAV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERDATAGRAMDATAV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERDATAGRAMDATAV4DISCARD")
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

// SetFWPMLAYERDATAGRAMDATAV6 sets the value of FWPMLAYERDATAGRAMDATAV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERDATAGRAMDATAV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERDATAGRAMDATAV6", (value))
}

// GetFWPMLAYERDATAGRAMDATAV6 gets the value of FWPMLAYERDATAGRAMDATAV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERDATAGRAMDATAV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERDATAGRAMDATAV6")
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

// SetFWPMLAYERDATAGRAMDATAV6DISCARD sets the value of FWPMLAYERDATAGRAMDATAV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERDATAGRAMDATAV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERDATAGRAMDATAV6DISCARD", (value))
}

// GetFWPMLAYERDATAGRAMDATAV6DISCARD gets the value of FWPMLAYERDATAGRAMDATAV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERDATAGRAMDATAV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERDATAGRAMDATAV6DISCARD")
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

// SetFWPMLAYEREGRESSVSWITCHETHERNET sets the value of FWPMLAYEREGRESSVSWITCHETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEREGRESSVSWITCHETHERNET(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEREGRESSVSWITCHETHERNET", (value))
}

// GetFWPMLAYEREGRESSVSWITCHETHERNET gets the value of FWPMLAYEREGRESSVSWITCHETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEREGRESSVSWITCHETHERNET() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEREGRESSVSWITCHETHERNET")
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

// SetFWPMLAYEREGRESSVSWITCHTRANSPORTV4 sets the value of FWPMLAYEREGRESSVSWITCHTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEREGRESSVSWITCHTRANSPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEREGRESSVSWITCHTRANSPORTV4", (value))
}

// GetFWPMLAYEREGRESSVSWITCHTRANSPORTV4 gets the value of FWPMLAYEREGRESSVSWITCHTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEREGRESSVSWITCHTRANSPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEREGRESSVSWITCHTRANSPORTV4")
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

// SetFWPMLAYEREGRESSVSWITCHTRANSPORTV6 sets the value of FWPMLAYEREGRESSVSWITCHTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEREGRESSVSWITCHTRANSPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEREGRESSVSWITCHTRANSPORTV6", (value))
}

// GetFWPMLAYEREGRESSVSWITCHTRANSPORTV6 gets the value of FWPMLAYEREGRESSVSWITCHTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEREGRESSVSWITCHTRANSPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEREGRESSVSWITCHTRANSPORTV6")
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

// SetFWPMLAYERIKEEXTV4 sets the value of FWPMLAYERIKEEXTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIKEEXTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIKEEXTV4", (value))
}

// GetFWPMLAYERIKEEXTV4 gets the value of FWPMLAYERIKEEXTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIKEEXTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIKEEXTV4")
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

// SetFWPMLAYERIKEEXTV6 sets the value of FWPMLAYERIKEEXTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIKEEXTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIKEEXTV6", (value))
}

// GetFWPMLAYERIKEEXTV6 gets the value of FWPMLAYERIKEEXTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIKEEXTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIKEEXTV6")
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

// SetFWPMLAYERINBOUNDICMPERRORV4 sets the value of FWPMLAYERINBOUNDICMPERRORV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDICMPERRORV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDICMPERRORV4", (value))
}

// GetFWPMLAYERINBOUNDICMPERRORV4 gets the value of FWPMLAYERINBOUNDICMPERRORV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDICMPERRORV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDICMPERRORV4")
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

// SetFWPMLAYERINBOUNDICMPERRORV4DISCARD sets the value of FWPMLAYERINBOUNDICMPERRORV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDICMPERRORV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDICMPERRORV4DISCARD", (value))
}

// GetFWPMLAYERINBOUNDICMPERRORV4DISCARD gets the value of FWPMLAYERINBOUNDICMPERRORV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDICMPERRORV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDICMPERRORV4DISCARD")
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

// SetFWPMLAYERINBOUNDICMPERRORV6 sets the value of FWPMLAYERINBOUNDICMPERRORV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDICMPERRORV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDICMPERRORV6", (value))
}

// GetFWPMLAYERINBOUNDICMPERRORV6 gets the value of FWPMLAYERINBOUNDICMPERRORV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDICMPERRORV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDICMPERRORV6")
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

// SetFWPMLAYERINBOUNDICMPERRORV6DISCARD sets the value of FWPMLAYERINBOUNDICMPERRORV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDICMPERRORV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDICMPERRORV6DISCARD", (value))
}

// GetFWPMLAYERINBOUNDICMPERRORV6DISCARD gets the value of FWPMLAYERINBOUNDICMPERRORV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDICMPERRORV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDICMPERRORV6DISCARD")
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

// SetFWPMLAYERINBOUNDIPPACKETV4 sets the value of FWPMLAYERINBOUNDIPPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDIPPACKETV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDIPPACKETV4", (value))
}

// GetFWPMLAYERINBOUNDIPPACKETV4 gets the value of FWPMLAYERINBOUNDIPPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDIPPACKETV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDIPPACKETV4")
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

// SetFWPMLAYERINBOUNDIPPACKETV4DISCARD sets the value of FWPMLAYERINBOUNDIPPACKETV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDIPPACKETV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDIPPACKETV4DISCARD", (value))
}

// GetFWPMLAYERINBOUNDIPPACKETV4DISCARD gets the value of FWPMLAYERINBOUNDIPPACKETV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDIPPACKETV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDIPPACKETV4DISCARD")
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

// SetFWPMLAYERINBOUNDIPPACKETV6 sets the value of FWPMLAYERINBOUNDIPPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDIPPACKETV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDIPPACKETV6", (value))
}

// GetFWPMLAYERINBOUNDIPPACKETV6 gets the value of FWPMLAYERINBOUNDIPPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDIPPACKETV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDIPPACKETV6")
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

// SetFWPMLAYERINBOUNDIPPACKETV6DISCARD sets the value of FWPMLAYERINBOUNDIPPACKETV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDIPPACKETV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDIPPACKETV6DISCARD", (value))
}

// GetFWPMLAYERINBOUNDIPPACKETV6DISCARD gets the value of FWPMLAYERINBOUNDIPPACKETV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDIPPACKETV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDIPPACKETV6DISCARD")
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

// SetFWPMLAYERINBOUNDMACFRAMEETHERNET sets the value of FWPMLAYERINBOUNDMACFRAMEETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDMACFRAMEETHERNET(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDMACFRAMEETHERNET", (value))
}

// GetFWPMLAYERINBOUNDMACFRAMEETHERNET gets the value of FWPMLAYERINBOUNDMACFRAMEETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDMACFRAMEETHERNET() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDMACFRAMEETHERNET")
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

// SetFWPMLAYERINBOUNDMACFRAMENATIVE sets the value of FWPMLAYERINBOUNDMACFRAMENATIVE for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDMACFRAMENATIVE(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDMACFRAMENATIVE", (value))
}

// GetFWPMLAYERINBOUNDMACFRAMENATIVE gets the value of FWPMLAYERINBOUNDMACFRAMENATIVE for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDMACFRAMENATIVE() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDMACFRAMENATIVE")
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

// SetFWPMLAYERINBOUNDMACFRAMENATIVEFAST sets the value of FWPMLAYERINBOUNDMACFRAMENATIVEFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDMACFRAMENATIVEFAST(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDMACFRAMENATIVEFAST", (value))
}

// GetFWPMLAYERINBOUNDMACFRAMENATIVEFAST gets the value of FWPMLAYERINBOUNDMACFRAMENATIVEFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDMACFRAMENATIVEFAST() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDMACFRAMENATIVEFAST")
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

// SetFWPMLAYERINBOUNDSSLTHROTTLING sets the value of FWPMLAYERINBOUNDSSLTHROTTLING for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDSSLTHROTTLING(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDSSLTHROTTLING", (value))
}

// GetFWPMLAYERINBOUNDSSLTHROTTLING gets the value of FWPMLAYERINBOUNDSSLTHROTTLING for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDSSLTHROTTLING() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDSSLTHROTTLING")
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

// SetFWPMLAYERINBOUNDTRANSPORTFAST sets the value of FWPMLAYERINBOUNDTRANSPORTFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDTRANSPORTFAST(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDTRANSPORTFAST", (value))
}

// GetFWPMLAYERINBOUNDTRANSPORTFAST gets the value of FWPMLAYERINBOUNDTRANSPORTFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDTRANSPORTFAST() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDTRANSPORTFAST")
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

// SetFWPMLAYERINBOUNDTRANSPORTV4 sets the value of FWPMLAYERINBOUNDTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDTRANSPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDTRANSPORTV4", (value))
}

// GetFWPMLAYERINBOUNDTRANSPORTV4 gets the value of FWPMLAYERINBOUNDTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDTRANSPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDTRANSPORTV4")
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

// SetFWPMLAYERINBOUNDTRANSPORTV4DISCARD sets the value of FWPMLAYERINBOUNDTRANSPORTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDTRANSPORTV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDTRANSPORTV4DISCARD", (value))
}

// GetFWPMLAYERINBOUNDTRANSPORTV4DISCARD gets the value of FWPMLAYERINBOUNDTRANSPORTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDTRANSPORTV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDTRANSPORTV4DISCARD")
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

// SetFWPMLAYERINBOUNDTRANSPORTV6 sets the value of FWPMLAYERINBOUNDTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDTRANSPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDTRANSPORTV6", (value))
}

// GetFWPMLAYERINBOUNDTRANSPORTV6 gets the value of FWPMLAYERINBOUNDTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDTRANSPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDTRANSPORTV6")
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

// SetFWPMLAYERINBOUNDTRANSPORTV6DISCARD sets the value of FWPMLAYERINBOUNDTRANSPORTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINBOUNDTRANSPORTV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINBOUNDTRANSPORTV6DISCARD", (value))
}

// GetFWPMLAYERINBOUNDTRANSPORTV6DISCARD gets the value of FWPMLAYERINBOUNDTRANSPORTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINBOUNDTRANSPORTV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINBOUNDTRANSPORTV6DISCARD")
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

// SetFWPMLAYERINGRESSVSWITCHETHERNET sets the value of FWPMLAYERINGRESSVSWITCHETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINGRESSVSWITCHETHERNET(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINGRESSVSWITCHETHERNET", (value))
}

// GetFWPMLAYERINGRESSVSWITCHETHERNET gets the value of FWPMLAYERINGRESSVSWITCHETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINGRESSVSWITCHETHERNET() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINGRESSVSWITCHETHERNET")
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

// SetFWPMLAYERINGRESSVSWITCHTRANSPORTV4 sets the value of FWPMLAYERINGRESSVSWITCHTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINGRESSVSWITCHTRANSPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINGRESSVSWITCHTRANSPORTV4", (value))
}

// GetFWPMLAYERINGRESSVSWITCHTRANSPORTV4 gets the value of FWPMLAYERINGRESSVSWITCHTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINGRESSVSWITCHTRANSPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINGRESSVSWITCHTRANSPORTV4")
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

// SetFWPMLAYERINGRESSVSWITCHTRANSPORTV6 sets the value of FWPMLAYERINGRESSVSWITCHTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERINGRESSVSWITCHTRANSPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERINGRESSVSWITCHTRANSPORTV6", (value))
}

// GetFWPMLAYERINGRESSVSWITCHTRANSPORTV6 gets the value of FWPMLAYERINGRESSVSWITCHTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERINGRESSVSWITCHTRANSPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERINGRESSVSWITCHTRANSPORTV6")
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

// SetFWPMLAYERIPFORWARDV4 sets the value of FWPMLAYERIPFORWARDV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPFORWARDV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPFORWARDV4", (value))
}

// GetFWPMLAYERIPFORWARDV4 gets the value of FWPMLAYERIPFORWARDV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPFORWARDV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPFORWARDV4")
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

// SetFWPMLAYERIPFORWARDV4DISCARD sets the value of FWPMLAYERIPFORWARDV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPFORWARDV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPFORWARDV4DISCARD", (value))
}

// GetFWPMLAYERIPFORWARDV4DISCARD gets the value of FWPMLAYERIPFORWARDV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPFORWARDV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPFORWARDV4DISCARD")
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

// SetFWPMLAYERIPFORWARDV6 sets the value of FWPMLAYERIPFORWARDV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPFORWARDV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPFORWARDV6", (value))
}

// GetFWPMLAYERIPFORWARDV6 gets the value of FWPMLAYERIPFORWARDV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPFORWARDV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPFORWARDV6")
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

// SetFWPMLAYERIPFORWARDV6DISCARD sets the value of FWPMLAYERIPFORWARDV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPFORWARDV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPFORWARDV6DISCARD", (value))
}

// GetFWPMLAYERIPFORWARDV6DISCARD gets the value of FWPMLAYERIPFORWARDV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPFORWARDV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPFORWARDV6DISCARD")
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

// SetFWPMLAYERIPSECKMDEMUXV4 sets the value of FWPMLAYERIPSECKMDEMUXV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPSECKMDEMUXV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPSECKMDEMUXV4", (value))
}

// GetFWPMLAYERIPSECKMDEMUXV4 gets the value of FWPMLAYERIPSECKMDEMUXV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPSECKMDEMUXV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPSECKMDEMUXV4")
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

// SetFWPMLAYERIPSECKMDEMUXV6 sets the value of FWPMLAYERIPSECKMDEMUXV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPSECKMDEMUXV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPSECKMDEMUXV6", (value))
}

// GetFWPMLAYERIPSECKMDEMUXV6 gets the value of FWPMLAYERIPSECKMDEMUXV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPSECKMDEMUXV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPSECKMDEMUXV6")
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

// SetFWPMLAYERIPSECV4 sets the value of FWPMLAYERIPSECV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPSECV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPSECV4", (value))
}

// GetFWPMLAYERIPSECV4 gets the value of FWPMLAYERIPSECV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPSECV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPSECV4")
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

// SetFWPMLAYERIPSECV6 sets the value of FWPMLAYERIPSECV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERIPSECV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERIPSECV6", (value))
}

// GetFWPMLAYERIPSECV6 gets the value of FWPMLAYERIPSECV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERIPSECV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERIPSECV6")
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

// SetFWPMLAYERKMAUTHORIZATION sets the value of FWPMLAYERKMAUTHORIZATION for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERKMAUTHORIZATION(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERKMAUTHORIZATION", (value))
}

// GetFWPMLAYERKMAUTHORIZATION gets the value of FWPMLAYERKMAUTHORIZATION for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERKMAUTHORIZATION() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERKMAUTHORIZATION")
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

// SetFWPMLAYERNAMERESOLUTIONCACHEV4 sets the value of FWPMLAYERNAMERESOLUTIONCACHEV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERNAMERESOLUTIONCACHEV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERNAMERESOLUTIONCACHEV4", (value))
}

// GetFWPMLAYERNAMERESOLUTIONCACHEV4 gets the value of FWPMLAYERNAMERESOLUTIONCACHEV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERNAMERESOLUTIONCACHEV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERNAMERESOLUTIONCACHEV4")
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

// SetFWPMLAYERNAMERESOLUTIONCACHEV6 sets the value of FWPMLAYERNAMERESOLUTIONCACHEV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERNAMERESOLUTIONCACHEV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERNAMERESOLUTIONCACHEV6", (value))
}

// GetFWPMLAYERNAMERESOLUTIONCACHEV6 gets the value of FWPMLAYERNAMERESOLUTIONCACHEV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERNAMERESOLUTIONCACHEV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERNAMERESOLUTIONCACHEV6")
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

// SetFWPMLAYEROUTBOUNDICMPERRORV4 sets the value of FWPMLAYEROUTBOUNDICMPERRORV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDICMPERRORV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDICMPERRORV4", (value))
}

// GetFWPMLAYEROUTBOUNDICMPERRORV4 gets the value of FWPMLAYEROUTBOUNDICMPERRORV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDICMPERRORV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDICMPERRORV4")
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

// SetFWPMLAYEROUTBOUNDICMPERRORV4DISCARD sets the value of FWPMLAYEROUTBOUNDICMPERRORV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDICMPERRORV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDICMPERRORV4DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDICMPERRORV4DISCARD gets the value of FWPMLAYEROUTBOUNDICMPERRORV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDICMPERRORV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDICMPERRORV4DISCARD")
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

// SetFWPMLAYEROUTBOUNDICMPERRORV6 sets the value of FWPMLAYEROUTBOUNDICMPERRORV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDICMPERRORV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDICMPERRORV6", (value))
}

// GetFWPMLAYEROUTBOUNDICMPERRORV6 gets the value of FWPMLAYEROUTBOUNDICMPERRORV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDICMPERRORV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDICMPERRORV6")
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

// SetFWPMLAYEROUTBOUNDICMPERRORV6DISCARD sets the value of FWPMLAYEROUTBOUNDICMPERRORV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDICMPERRORV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDICMPERRORV6DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDICMPERRORV6DISCARD gets the value of FWPMLAYEROUTBOUNDICMPERRORV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDICMPERRORV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDICMPERRORV6DISCARD")
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

// SetFWPMLAYEROUTBOUNDIPPACKETV4 sets the value of FWPMLAYEROUTBOUNDIPPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDIPPACKETV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDIPPACKETV4", (value))
}

// GetFWPMLAYEROUTBOUNDIPPACKETV4 gets the value of FWPMLAYEROUTBOUNDIPPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDIPPACKETV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDIPPACKETV4")
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

// SetFWPMLAYEROUTBOUNDIPPACKETV4DISCARD sets the value of FWPMLAYEROUTBOUNDIPPACKETV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDIPPACKETV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDIPPACKETV4DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDIPPACKETV4DISCARD gets the value of FWPMLAYEROUTBOUNDIPPACKETV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDIPPACKETV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDIPPACKETV4DISCARD")
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

// SetFWPMLAYEROUTBOUNDIPPACKETV6 sets the value of FWPMLAYEROUTBOUNDIPPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDIPPACKETV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDIPPACKETV6", (value))
}

// GetFWPMLAYEROUTBOUNDIPPACKETV6 gets the value of FWPMLAYEROUTBOUNDIPPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDIPPACKETV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDIPPACKETV6")
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

// SetFWPMLAYEROUTBOUNDIPPACKETV6DISCARD sets the value of FWPMLAYEROUTBOUNDIPPACKETV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDIPPACKETV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDIPPACKETV6DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDIPPACKETV6DISCARD gets the value of FWPMLAYEROUTBOUNDIPPACKETV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDIPPACKETV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDIPPACKETV6DISCARD")
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

// SetFWPMLAYEROUTBOUNDMACFRAMEETHERNET sets the value of FWPMLAYEROUTBOUNDMACFRAMEETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDMACFRAMEETHERNET(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDMACFRAMEETHERNET", (value))
}

// GetFWPMLAYEROUTBOUNDMACFRAMEETHERNET gets the value of FWPMLAYEROUTBOUNDMACFRAMEETHERNET for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDMACFRAMEETHERNET() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDMACFRAMEETHERNET")
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

// SetFWPMLAYEROUTBOUNDMACFRAMENATIVE sets the value of FWPMLAYEROUTBOUNDMACFRAMENATIVE for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDMACFRAMENATIVE(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDMACFRAMENATIVE", (value))
}

// GetFWPMLAYEROUTBOUNDMACFRAMENATIVE gets the value of FWPMLAYEROUTBOUNDMACFRAMENATIVE for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDMACFRAMENATIVE() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDMACFRAMENATIVE")
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

// SetFWPMLAYEROUTBOUNDMACFRAMENATIVEFAST sets the value of FWPMLAYEROUTBOUNDMACFRAMENATIVEFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDMACFRAMENATIVEFAST(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDMACFRAMENATIVEFAST", (value))
}

// GetFWPMLAYEROUTBOUNDMACFRAMENATIVEFAST gets the value of FWPMLAYEROUTBOUNDMACFRAMENATIVEFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDMACFRAMENATIVEFAST() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDMACFRAMENATIVEFAST")
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

// SetFWPMLAYEROUTBOUNDTRANSPORTFAST sets the value of FWPMLAYEROUTBOUNDTRANSPORTFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDTRANSPORTFAST(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDTRANSPORTFAST", (value))
}

// GetFWPMLAYEROUTBOUNDTRANSPORTFAST gets the value of FWPMLAYEROUTBOUNDTRANSPORTFAST for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDTRANSPORTFAST() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDTRANSPORTFAST")
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

// SetFWPMLAYEROUTBOUNDTRANSPORTV4 sets the value of FWPMLAYEROUTBOUNDTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDTRANSPORTV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDTRANSPORTV4", (value))
}

// GetFWPMLAYEROUTBOUNDTRANSPORTV4 gets the value of FWPMLAYEROUTBOUNDTRANSPORTV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDTRANSPORTV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDTRANSPORTV4")
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

// SetFWPMLAYEROUTBOUNDTRANSPORTV4DISCARD sets the value of FWPMLAYEROUTBOUNDTRANSPORTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDTRANSPORTV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDTRANSPORTV4DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDTRANSPORTV4DISCARD gets the value of FWPMLAYEROUTBOUNDTRANSPORTV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDTRANSPORTV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDTRANSPORTV4DISCARD")
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

// SetFWPMLAYEROUTBOUNDTRANSPORTV6 sets the value of FWPMLAYEROUTBOUNDTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDTRANSPORTV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDTRANSPORTV6", (value))
}

// GetFWPMLAYEROUTBOUNDTRANSPORTV6 gets the value of FWPMLAYEROUTBOUNDTRANSPORTV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDTRANSPORTV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDTRANSPORTV6")
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

// SetFWPMLAYEROUTBOUNDTRANSPORTV6DISCARD sets the value of FWPMLAYEROUTBOUNDTRANSPORTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYEROUTBOUNDTRANSPORTV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYEROUTBOUNDTRANSPORTV6DISCARD", (value))
}

// GetFWPMLAYEROUTBOUNDTRANSPORTV6DISCARD gets the value of FWPMLAYEROUTBOUNDTRANSPORTV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYEROUTBOUNDTRANSPORTV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYEROUTBOUNDTRANSPORTV6DISCARD")
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

// SetFWPMLAYERRPCEPADD sets the value of FWPMLAYERRPCEPADD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERRPCEPADD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERRPCEPADD", (value))
}

// GetFWPMLAYERRPCEPADD gets the value of FWPMLAYERRPCEPADD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERRPCEPADD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERRPCEPADD")
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

// SetFWPMLAYERRPCEPMAP sets the value of FWPMLAYERRPCEPMAP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERRPCEPMAP(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERRPCEPMAP", (value))
}

// GetFWPMLAYERRPCEPMAP gets the value of FWPMLAYERRPCEPMAP for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERRPCEPMAP() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERRPCEPMAP")
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

// SetFWPMLAYERRPCPROXYCONN sets the value of FWPMLAYERRPCPROXYCONN for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERRPCPROXYCONN(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERRPCPROXYCONN", (value))
}

// GetFWPMLAYERRPCPROXYCONN gets the value of FWPMLAYERRPCPROXYCONN for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERRPCPROXYCONN() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERRPCPROXYCONN")
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

// SetFWPMLAYERRPCPROXYIF sets the value of FWPMLAYERRPCPROXYIF for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERRPCPROXYIF(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERRPCPROXYIF", (value))
}

// GetFWPMLAYERRPCPROXYIF gets the value of FWPMLAYERRPCPROXYIF for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERRPCPROXYIF() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERRPCPROXYIF")
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

// SetFWPMLAYERRPCUM sets the value of FWPMLAYERRPCUM for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERRPCUM(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERRPCUM", (value))
}

// GetFWPMLAYERRPCUM gets the value of FWPMLAYERRPCUM for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERRPCUM() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERRPCUM")
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

// SetFWPMLAYERSTREAMPACKETV4 sets the value of FWPMLAYERSTREAMPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMPACKETV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMPACKETV4", (value))
}

// GetFWPMLAYERSTREAMPACKETV4 gets the value of FWPMLAYERSTREAMPACKETV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMPACKETV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMPACKETV4")
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

// SetFWPMLAYERSTREAMPACKETV6 sets the value of FWPMLAYERSTREAMPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMPACKETV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMPACKETV6", (value))
}

// GetFWPMLAYERSTREAMPACKETV6 gets the value of FWPMLAYERSTREAMPACKETV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMPACKETV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMPACKETV6")
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

// SetFWPMLAYERSTREAMV4 sets the value of FWPMLAYERSTREAMV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMV4(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMV4", (value))
}

// GetFWPMLAYERSTREAMV4 gets the value of FWPMLAYERSTREAMV4 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMV4() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMV4")
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

// SetFWPMLAYERSTREAMV4DISCARD sets the value of FWPMLAYERSTREAMV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMV4DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMV4DISCARD", (value))
}

// GetFWPMLAYERSTREAMV4DISCARD gets the value of FWPMLAYERSTREAMV4DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMV4DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMV4DISCARD")
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

// SetFWPMLAYERSTREAMV6 sets the value of FWPMLAYERSTREAMV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMV6(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMV6", (value))
}

// GetFWPMLAYERSTREAMV6 gets the value of FWPMLAYERSTREAMV6 for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMV6() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMV6")
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

// SetFWPMLAYERSTREAMV6DISCARD sets the value of FWPMLAYERSTREAMV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyFWPMLAYERSTREAMV6DISCARD(value uint64) (err error) {
	return instance.SetProperty("FWPMLAYERSTREAMV6DISCARD", (value))
}

// GetFWPMLAYERSTREAMV6DISCARD gets the value of FWPMLAYERSTREAMV6DISCARD for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyFWPMLAYERSTREAMV6DISCARD() (value uint64, err error) {
	retValue, err := instance.GetProperty("FWPMLAYERSTREAMV6DISCARD")
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

// SetTotal sets the value of Total for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) SetPropertyTotal(value uint64) (err error) {
	return instance.SetProperty("Total", (value))
}

// GetTotal gets the value of Total for the instance
func (instance *Win32_PerfFormattedData_Counters_WFPClassify) GetPropertyTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("Total")
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
