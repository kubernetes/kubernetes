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

// MSFT_FileStorageTier struct
type MSFT_FileStorageTier struct {
	*cim.WmiInstance

	//
	DesiredStorageTierClass uint16

	//
	DesiredStorageTierName string

	//
	FilePath string

	//
	FileSize uint64

	//
	FileSizeOnDesiredStorageTier uint64

	//
	FileSizeOnDesiredStorageTierClass uint64

	//
	PlacementStatus uint16

	//
	State uint16
}

func NewMSFT_FileStorageTierEx1(instance *cim.WmiInstance) (newInstance *MSFT_FileStorageTier, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_FileStorageTier{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_FileStorageTierEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_FileStorageTier, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_FileStorageTier{
		WmiInstance: tmp,
	}
	return
}

// SetDesiredStorageTierClass sets the value of DesiredStorageTierClass for the instance
func (instance *MSFT_FileStorageTier) SetPropertyDesiredStorageTierClass(value uint16) (err error) {
	return instance.SetProperty("DesiredStorageTierClass", (value))
}

// GetDesiredStorageTierClass gets the value of DesiredStorageTierClass for the instance
func (instance *MSFT_FileStorageTier) GetPropertyDesiredStorageTierClass() (value uint16, err error) {
	retValue, err := instance.GetProperty("DesiredStorageTierClass")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetDesiredStorageTierName sets the value of DesiredStorageTierName for the instance
func (instance *MSFT_FileStorageTier) SetPropertyDesiredStorageTierName(value string) (err error) {
	return instance.SetProperty("DesiredStorageTierName", (value))
}

// GetDesiredStorageTierName gets the value of DesiredStorageTierName for the instance
func (instance *MSFT_FileStorageTier) GetPropertyDesiredStorageTierName() (value string, err error) {
	retValue, err := instance.GetProperty("DesiredStorageTierName")
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

// SetFilePath sets the value of FilePath for the instance
func (instance *MSFT_FileStorageTier) SetPropertyFilePath(value string) (err error) {
	return instance.SetProperty("FilePath", (value))
}

// GetFilePath gets the value of FilePath for the instance
func (instance *MSFT_FileStorageTier) GetPropertyFilePath() (value string, err error) {
	retValue, err := instance.GetProperty("FilePath")
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

// SetFileSize sets the value of FileSize for the instance
func (instance *MSFT_FileStorageTier) SetPropertyFileSize(value uint64) (err error) {
	return instance.SetProperty("FileSize", (value))
}

// GetFileSize gets the value of FileSize for the instance
func (instance *MSFT_FileStorageTier) GetPropertyFileSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileSize")
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

// SetFileSizeOnDesiredStorageTier sets the value of FileSizeOnDesiredStorageTier for the instance
func (instance *MSFT_FileStorageTier) SetPropertyFileSizeOnDesiredStorageTier(value uint64) (err error) {
	return instance.SetProperty("FileSizeOnDesiredStorageTier", (value))
}

// GetFileSizeOnDesiredStorageTier gets the value of FileSizeOnDesiredStorageTier for the instance
func (instance *MSFT_FileStorageTier) GetPropertyFileSizeOnDesiredStorageTier() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileSizeOnDesiredStorageTier")
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

// SetFileSizeOnDesiredStorageTierClass sets the value of FileSizeOnDesiredStorageTierClass for the instance
func (instance *MSFT_FileStorageTier) SetPropertyFileSizeOnDesiredStorageTierClass(value uint64) (err error) {
	return instance.SetProperty("FileSizeOnDesiredStorageTierClass", (value))
}

// GetFileSizeOnDesiredStorageTierClass gets the value of FileSizeOnDesiredStorageTierClass for the instance
func (instance *MSFT_FileStorageTier) GetPropertyFileSizeOnDesiredStorageTierClass() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileSizeOnDesiredStorageTierClass")
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

// SetPlacementStatus sets the value of PlacementStatus for the instance
func (instance *MSFT_FileStorageTier) SetPropertyPlacementStatus(value uint16) (err error) {
	return instance.SetProperty("PlacementStatus", (value))
}

// GetPlacementStatus gets the value of PlacementStatus for the instance
func (instance *MSFT_FileStorageTier) GetPropertyPlacementStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("PlacementStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetState sets the value of State for the instance
func (instance *MSFT_FileStorageTier) SetPropertyState(value uint16) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *MSFT_FileStorageTier) GetPropertyState() (value uint16, err error) {
	retValue, err := instance.GetProperty("State")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

//

// <param name="FilePath" type="string "></param>
// <param name="Volume" type="MSFT_Volume "></param>
// <param name="VolumeDriveLetter" type="byte "></param>
// <param name="VolumePath" type="string "></param>

// <param name="FileStorageTier" type="MSFT_FileStorageTier []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileStorageTier) Get( /* IN */ FilePath string,
	/* IN */ VolumeDriveLetter byte,
	/* IN */ VolumePath string,
	/* IN */ Volume MSFT_Volume,
	/* OUT */ FileStorageTier []MSFT_FileStorageTier) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Get", FilePath, VolumeDriveLetter, VolumePath, Volume)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DesiredStorageTier" type="MSFT_StorageTier "></param>
// <param name="DesiredStorageTierClass" type="uint16 "></param>
// <param name="DesiredStorageTierFriendlyName" type="string "></param>
// <param name="DesiredStorageTierUniqueId" type="string "></param>
// <param name="FilePath" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileStorageTier) Set( /* IN */ FilePath string,
	/* IN */ DesiredStorageTierFriendlyName string,
	/* IN */ DesiredStorageTierUniqueId string,
	/* IN */ DesiredStorageTierClass uint16,
	/* IN */ DesiredStorageTier MSFT_StorageTier) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Set", FilePath, DesiredStorageTierFriendlyName, DesiredStorageTierUniqueId, DesiredStorageTierClass, DesiredStorageTier)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="FilePath" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileStorageTier) Clear( /* IN */ FilePath string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Clear", FilePath)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
