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

// MSFT_FileServerToVolume struct
type MSFT_FileServerToVolume struct {
	*cim.WmiInstance

	//
	FileServer MSFT_FileServer

	//
	Volume MSFT_Volume
}

func NewMSFT_FileServerToVolumeEx1(instance *cim.WmiInstance) (newInstance *MSFT_FileServerToVolume, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServerToVolume{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_FileServerToVolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_FileServerToVolume, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServerToVolume{
		WmiInstance: tmp,
	}
	return
}

// SetFileServer sets the value of FileServer for the instance
func (instance *MSFT_FileServerToVolume) SetPropertyFileServer(value MSFT_FileServer) (err error) {
	return instance.SetProperty("FileServer", (value))
}

// GetFileServer gets the value of FileServer for the instance
func (instance *MSFT_FileServerToVolume) GetPropertyFileServer() (value MSFT_FileServer, err error) {
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

// SetVolume sets the value of Volume for the instance
func (instance *MSFT_FileServerToVolume) SetPropertyVolume(value MSFT_Volume) (err error) {
	return instance.SetProperty("Volume", (value))
}

// GetVolume gets the value of Volume for the instance
func (instance *MSFT_FileServerToVolume) GetPropertyVolume() (value MSFT_Volume, err error) {
	retValue, err := instance.GetProperty("Volume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_Volume)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_Volume is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_Volume(valuetmp)

	return
}
