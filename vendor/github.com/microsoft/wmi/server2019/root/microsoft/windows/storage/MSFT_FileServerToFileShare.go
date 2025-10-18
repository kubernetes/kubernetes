// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_FileServerToFileShare struct
type MSFT_FileServerToFileShare struct {
	*cim.WmiInstance

	//
	FileServer MSFT_FileServer

	//
	FileShare MSFT_FileShare
}

func NewMSFT_FileServerToFileShareEx1(instance *cim.WmiInstance) (newInstance *MSFT_FileServerToFileShare, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServerToFileShare{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_FileServerToFileShareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_FileServerToFileShare, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServerToFileShare{
		WmiInstance: tmp,
	}
	return
}

// SetFileServer sets the value of FileServer for the instance
func (instance *MSFT_FileServerToFileShare) SetPropertyFileServer(value MSFT_FileServer) (err error) {
	return instance.SetProperty("FileServer", (value))
}

// GetFileServer gets the value of FileServer for the instance
func (instance *MSFT_FileServerToFileShare) GetPropertyFileServer() (value MSFT_FileServer, err error) {
	retValue, err := instance.GetProperty("FileServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_FileServer)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_FileServer is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_FileServer(valuetmp)

	return
}

// SetFileShare sets the value of FileShare for the instance
func (instance *MSFT_FileServerToFileShare) SetPropertyFileShare(value MSFT_FileShare) (err error) {
	return instance.SetProperty("FileShare", (value))
}

// GetFileShare gets the value of FileShare for the instance
func (instance *MSFT_FileServerToFileShare) GetPropertyFileShare() (value MSFT_FileShare, err error) {
	retValue, err := instance.GetProperty("FileShare")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_FileShare)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_FileShare is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_FileShare(valuetmp)

	return
}
