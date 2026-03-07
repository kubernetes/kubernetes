// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_FileServer struct
type MSFT_FileServer struct {
	*MSFT_StorageObject

	//
	FileSharingProtocols []uint16

	//
	FileSharingProtocolVersions []string

	//
	FriendlyName string

	//
	HealthStatus uint16

	//
	HostNames []string

	//
	OperationalStatus []uint16

	//
	OtherOperationalStatusDescription string

	//
	SupportsContinuouslyAvailableFileShare bool

	//
	SupportsFileShareCreation bool
}

func NewMSFT_FileServerEx1(instance *cim.WmiInstance) (newInstance *MSFT_FileServer, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServer{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_FileServerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_FileServer, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_FileServer{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetFileSharingProtocols sets the value of FileSharingProtocols for the instance
func (instance *MSFT_FileServer) SetPropertyFileSharingProtocols(value []uint16) (err error) {
	return instance.SetProperty("FileSharingProtocols", (value))
}

// GetFileSharingProtocols gets the value of FileSharingProtocols for the instance
func (instance *MSFT_FileServer) GetPropertyFileSharingProtocols() (value []uint16, err error) {
	retValue, err := instance.GetProperty("FileSharingProtocols")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetFileSharingProtocolVersions sets the value of FileSharingProtocolVersions for the instance
func (instance *MSFT_FileServer) SetPropertyFileSharingProtocolVersions(value []string) (err error) {
	return instance.SetProperty("FileSharingProtocolVersions", (value))
}

// GetFileSharingProtocolVersions gets the value of FileSharingProtocolVersions for the instance
func (instance *MSFT_FileServer) GetPropertyFileSharingProtocolVersions() (value []string, err error) {
	retValue, err := instance.GetProperty("FileSharingProtocolVersions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *MSFT_FileServer) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_FileServer) GetPropertyFriendlyName() (value string, err error) {
	retValue, err := instance.GetProperty("FriendlyName")
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_FileServer) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_FileServer) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
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

// SetHostNames sets the value of HostNames for the instance
func (instance *MSFT_FileServer) SetPropertyHostNames(value []string) (err error) {
	return instance.SetProperty("HostNames", (value))
}

// GetHostNames gets the value of HostNames for the instance
func (instance *MSFT_FileServer) GetPropertyHostNames() (value []string, err error) {
	retValue, err := instance.GetProperty("HostNames")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_FileServer) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_FileServer) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetOtherOperationalStatusDescription sets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_FileServer) SetPropertyOtherOperationalStatusDescription(value string) (err error) {
	return instance.SetProperty("OtherOperationalStatusDescription", (value))
}

// GetOtherOperationalStatusDescription gets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_FileServer) GetPropertyOtherOperationalStatusDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherOperationalStatusDescription")
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

// SetSupportsContinuouslyAvailableFileShare sets the value of SupportsContinuouslyAvailableFileShare for the instance
func (instance *MSFT_FileServer) SetPropertySupportsContinuouslyAvailableFileShare(value bool) (err error) {
	return instance.SetProperty("SupportsContinuouslyAvailableFileShare", (value))
}

// GetSupportsContinuouslyAvailableFileShare gets the value of SupportsContinuouslyAvailableFileShare for the instance
func (instance *MSFT_FileServer) GetPropertySupportsContinuouslyAvailableFileShare() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsContinuouslyAvailableFileShare")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSupportsFileShareCreation sets the value of SupportsFileShareCreation for the instance
func (instance *MSFT_FileServer) SetPropertySupportsFileShareCreation(value bool) (err error) {
	return instance.SetProperty("SupportsFileShareCreation", (value))
}

// GetSupportsFileShareCreation gets the value of SupportsFileShareCreation for the instance
func (instance *MSFT_FileServer) GetPropertySupportsFileShareCreation() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsFileShareCreation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

//

// <param name="ContinuouslyAvailable" type="bool "></param>
// <param name="Description" type="string "></param>
// <param name="EncryptData" type="bool "></param>
// <param name="FileSharingProtocol" type="uint16 "></param>
// <param name="Name" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="SourceVolume" type="MSFT_Volume "></param>
// <param name="VolumeRelativePath" type="string "></param>

// <param name="CreatedFileShare" type="MSFT_FileShare "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileServer) CreateFileShare( /* IN */ Name string,
	/* IN */ Description string,
	/* IN */ SourceVolume MSFT_Volume,
	/* IN */ VolumeRelativePath string,
	/* IN */ ContinuouslyAvailable bool,
	/* IN */ EncryptData bool,
	/* IN */ FileSharingProtocol uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedFileShare MSFT_FileShare,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateFileShare", Name, Description, SourceVolume, VolumeRelativePath, ContinuouslyAvailable, EncryptData, FileSharingProtocol, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileServer) DeleteObject( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_FileServer) SetFriendlyName( /* IN */ FriendlyName string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetFriendlyName", FriendlyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
