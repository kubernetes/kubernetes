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

// Win32_NTEventlogFile struct
type Win32_NTEventlogFile struct {
	*CIM_DataFile

	//
	LogfileName string

	//
	MaxFileSize uint32

	//
	NumberOfRecords uint32

	//
	OverwriteOutDated uint32

	//
	OverWritePolicy string

	//
	Sources []string
}

func NewWin32_NTEventlogFileEx1(instance *cim.WmiInstance) (newInstance *Win32_NTEventlogFile, err error) {
	tmp, err := NewCIM_DataFileEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NTEventlogFile{
		CIM_DataFile: tmp,
	}
	return
}

func NewWin32_NTEventlogFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NTEventlogFile, err error) {
	tmp, err := NewCIM_DataFileEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NTEventlogFile{
		CIM_DataFile: tmp,
	}
	return
}

// SetLogfileName sets the value of LogfileName for the instance
func (instance *Win32_NTEventlogFile) SetPropertyLogfileName(value string) (err error) {
	return instance.SetProperty("LogfileName", (value))
}

// GetLogfileName gets the value of LogfileName for the instance
func (instance *Win32_NTEventlogFile) GetPropertyLogfileName() (value string, err error) {
	retValue, err := instance.GetProperty("LogfileName")
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

// SetMaxFileSize sets the value of MaxFileSize for the instance
func (instance *Win32_NTEventlogFile) SetPropertyMaxFileSize(value uint32) (err error) {
	return instance.SetProperty("MaxFileSize", (value))
}

// GetMaxFileSize gets the value of MaxFileSize for the instance
func (instance *Win32_NTEventlogFile) GetPropertyMaxFileSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxFileSize")
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

// SetNumberOfRecords sets the value of NumberOfRecords for the instance
func (instance *Win32_NTEventlogFile) SetPropertyNumberOfRecords(value uint32) (err error) {
	return instance.SetProperty("NumberOfRecords", (value))
}

// GetNumberOfRecords gets the value of NumberOfRecords for the instance
func (instance *Win32_NTEventlogFile) GetPropertyNumberOfRecords() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfRecords")
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

// SetOverwriteOutDated sets the value of OverwriteOutDated for the instance
func (instance *Win32_NTEventlogFile) SetPropertyOverwriteOutDated(value uint32) (err error) {
	return instance.SetProperty("OverwriteOutDated", (value))
}

// GetOverwriteOutDated gets the value of OverwriteOutDated for the instance
func (instance *Win32_NTEventlogFile) GetPropertyOverwriteOutDated() (value uint32, err error) {
	retValue, err := instance.GetProperty("OverwriteOutDated")
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

// SetOverWritePolicy sets the value of OverWritePolicy for the instance
func (instance *Win32_NTEventlogFile) SetPropertyOverWritePolicy(value string) (err error) {
	return instance.SetProperty("OverWritePolicy", (value))
}

// GetOverWritePolicy gets the value of OverWritePolicy for the instance
func (instance *Win32_NTEventlogFile) GetPropertyOverWritePolicy() (value string, err error) {
	retValue, err := instance.GetProperty("OverWritePolicy")
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

// SetSources sets the value of Sources for the instance
func (instance *Win32_NTEventlogFile) SetPropertySources(value []string) (err error) {
	return instance.SetProperty("Sources", (value))
}

// GetSources gets the value of Sources for the instance
func (instance *Win32_NTEventlogFile) GetPropertySources() (value []string, err error) {
	retValue, err := instance.GetProperty("Sources")
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

//

// <param name="ArchiveFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NTEventlogFile) ClearEventlog( /* IN */ ArchiveFileName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ClearEventlog", ArchiveFileName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ArchiveFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NTEventlogFile) BackupEventlog( /* IN */ ArchiveFileName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("BackupEventlog", ArchiveFileName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
