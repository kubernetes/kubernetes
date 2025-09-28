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

// CIM_FileSpecification struct
type CIM_FileSpecification struct {
	*CIM_Check

	//
	CheckSum uint32

	//
	CRC1 uint32

	//
	CRC2 uint32

	//
	CreateTimeStamp string

	//
	FileSize uint64

	//
	MD5Checksum string
}

func NewCIM_FileSpecificationEx1(instance *cim.WmiInstance) (newInstance *CIM_FileSpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_FileSpecification{
		CIM_Check: tmp,
	}
	return
}

func NewCIM_FileSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FileSpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FileSpecification{
		CIM_Check: tmp,
	}
	return
}

// SetCheckSum sets the value of CheckSum for the instance
func (instance *CIM_FileSpecification) SetPropertyCheckSum(value uint32) (err error) {
	return instance.SetProperty("CheckSum", (value))
}

// GetCheckSum gets the value of CheckSum for the instance
func (instance *CIM_FileSpecification) GetPropertyCheckSum() (value uint32, err error) {
	retValue, err := instance.GetProperty("CheckSum")
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

// SetCRC1 sets the value of CRC1 for the instance
func (instance *CIM_FileSpecification) SetPropertyCRC1(value uint32) (err error) {
	return instance.SetProperty("CRC1", (value))
}

// GetCRC1 gets the value of CRC1 for the instance
func (instance *CIM_FileSpecification) GetPropertyCRC1() (value uint32, err error) {
	retValue, err := instance.GetProperty("CRC1")
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

// SetCRC2 sets the value of CRC2 for the instance
func (instance *CIM_FileSpecification) SetPropertyCRC2(value uint32) (err error) {
	return instance.SetProperty("CRC2", (value))
}

// GetCRC2 gets the value of CRC2 for the instance
func (instance *CIM_FileSpecification) GetPropertyCRC2() (value uint32, err error) {
	retValue, err := instance.GetProperty("CRC2")
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

// SetCreateTimeStamp sets the value of CreateTimeStamp for the instance
func (instance *CIM_FileSpecification) SetPropertyCreateTimeStamp(value string) (err error) {
	return instance.SetProperty("CreateTimeStamp", (value))
}

// GetCreateTimeStamp gets the value of CreateTimeStamp for the instance
func (instance *CIM_FileSpecification) GetPropertyCreateTimeStamp() (value string, err error) {
	retValue, err := instance.GetProperty("CreateTimeStamp")
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
func (instance *CIM_FileSpecification) SetPropertyFileSize(value uint64) (err error) {
	return instance.SetProperty("FileSize", (value))
}

// GetFileSize gets the value of FileSize for the instance
func (instance *CIM_FileSpecification) GetPropertyFileSize() (value uint64, err error) {
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

// SetMD5Checksum sets the value of MD5Checksum for the instance
func (instance *CIM_FileSpecification) SetPropertyMD5Checksum(value string) (err error) {
	return instance.SetProperty("MD5Checksum", (value))
}

// GetMD5Checksum gets the value of MD5Checksum for the instance
func (instance *CIM_FileSpecification) GetPropertyMD5Checksum() (value string, err error) {
	retValue, err := instance.GetProperty("MD5Checksum")
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
