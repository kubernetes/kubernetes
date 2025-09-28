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

// MSFT_VolumeToFileShare struct
type MSFT_VolumeToFileShare struct {
	*cim.WmiInstance

	//
	FileShare MSFT_FileShare

	//
	Volume MSFT_Volume
}

func NewMSFT_VolumeToFileShareEx1(instance *cim.WmiInstance) (newInstance *MSFT_VolumeToFileShare, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_VolumeToFileShare{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_VolumeToFileShareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VolumeToFileShare, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VolumeToFileShare{
		WmiInstance: tmp,
	}
	return
}

// SetFileShare sets the value of FileShare for the instance
func (instance *MSFT_VolumeToFileShare) SetPropertyFileShare(value MSFT_FileShare) (err error) {
	return instance.SetProperty("FileShare", (value))
}

// GetFileShare gets the value of FileShare for the instance
func (instance *MSFT_VolumeToFileShare) GetPropertyFileShare() (value MSFT_FileShare, err error) {
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

// SetVolume sets the value of Volume for the instance
func (instance *MSFT_VolumeToFileShare) SetPropertyVolume(value MSFT_Volume) (err error) {
	return instance.SetProperty("Volume", (value))
}

// GetVolume gets the value of Volume for the instance
func (instance *MSFT_VolumeToFileShare) GetPropertyVolume() (value MSFT_Volume, err error) {
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
