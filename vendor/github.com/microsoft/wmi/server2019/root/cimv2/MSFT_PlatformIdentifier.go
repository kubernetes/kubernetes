// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// MSFT_PlatformIdentifier struct
type MSFT_PlatformIdentifier struct {
	*cim.WmiInstance
}

func NewMSFT_PlatformIdentifierEx1(instance *cim.WmiInstance) (newInstance *MSFT_PlatformIdentifier, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_PlatformIdentifier{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_PlatformIdentifierEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_PlatformIdentifier, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_PlatformIdentifier{
		WmiInstance: tmp,
	}
	return
}

//

// <param name="Name" type="string "></param>

// <param name="Identifier" type="string "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PlatformIdentifier) GetPlatformIdentifier( /* IN */ Name string,
	/* OUT */ Identifier string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetPlatformIdentifier", Name)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
