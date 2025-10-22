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

// CIM_LogicalFile struct
type CIM_LogicalFile struct {
	*CIM_LogicalElement

	//
	AccessMask uint32

	//
	Archive bool

	//
	Compressed bool

	//
	CompressionMethod string

	//
	CreationClassName string

	//
	CreationDate string

	//
	CSCreationClassName string

	//
	CSName string

	//
	Drive string

	//
	EightDotThreeFileName string

	//
	Encrypted bool

	//
	EncryptionMethod string

	//
	Extension string

	//
	FileName string

	//
	FileSize uint64

	//
	FileType string

	//
	FSCreationClassName string

	//
	FSName string

	//
	Hidden bool

	//
	InUseCount uint64

	//
	LastAccessed string

	//
	LastModified string

	//
	Path string

	//
	Readable bool

	//
	System bool

	//
	Writeable bool
}

func NewCIM_LogicalFileEx1(instance *cim.WmiInstance) (newInstance *CIM_LogicalFile, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalFile{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_LogicalFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_LogicalFile, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalFile{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetAccessMask sets the value of AccessMask for the instance
func (instance *CIM_LogicalFile) SetPropertyAccessMask(value uint32) (err error) {
	return instance.SetProperty("AccessMask", (value))
}

// GetAccessMask gets the value of AccessMask for the instance
func (instance *CIM_LogicalFile) GetPropertyAccessMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("AccessMask")
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

// SetArchive sets the value of Archive for the instance
func (instance *CIM_LogicalFile) SetPropertyArchive(value bool) (err error) {
	return instance.SetProperty("Archive", (value))
}

// GetArchive gets the value of Archive for the instance
func (instance *CIM_LogicalFile) GetPropertyArchive() (value bool, err error) {
	retValue, err := instance.GetProperty("Archive")
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

// SetCompressed sets the value of Compressed for the instance
func (instance *CIM_LogicalFile) SetPropertyCompressed(value bool) (err error) {
	return instance.SetProperty("Compressed", (value))
}

// GetCompressed gets the value of Compressed for the instance
func (instance *CIM_LogicalFile) GetPropertyCompressed() (value bool, err error) {
	retValue, err := instance.GetProperty("Compressed")
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

// SetCompressionMethod sets the value of CompressionMethod for the instance
func (instance *CIM_LogicalFile) SetPropertyCompressionMethod(value string) (err error) {
	return instance.SetProperty("CompressionMethod", (value))
}

// GetCompressionMethod gets the value of CompressionMethod for the instance
func (instance *CIM_LogicalFile) GetPropertyCompressionMethod() (value string, err error) {
	retValue, err := instance.GetProperty("CompressionMethod")
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

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_LogicalFile) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_LogicalFile) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCreationDate sets the value of CreationDate for the instance
func (instance *CIM_LogicalFile) SetPropertyCreationDate(value string) (err error) {
	return instance.SetProperty("CreationDate", (value))
}

// GetCreationDate gets the value of CreationDate for the instance
func (instance *CIM_LogicalFile) GetPropertyCreationDate() (value string, err error) {
	retValue, err := instance.GetProperty("CreationDate")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_LogicalFile) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_LogicalFile) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_LogicalFile) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_LogicalFile) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetDrive sets the value of Drive for the instance
func (instance *CIM_LogicalFile) SetPropertyDrive(value string) (err error) {
	return instance.SetProperty("Drive", (value))
}

// GetDrive gets the value of Drive for the instance
func (instance *CIM_LogicalFile) GetPropertyDrive() (value string, err error) {
	retValue, err := instance.GetProperty("Drive")
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

// SetEightDotThreeFileName sets the value of EightDotThreeFileName for the instance
func (instance *CIM_LogicalFile) SetPropertyEightDotThreeFileName(value string) (err error) {
	return instance.SetProperty("EightDotThreeFileName", (value))
}

// GetEightDotThreeFileName gets the value of EightDotThreeFileName for the instance
func (instance *CIM_LogicalFile) GetPropertyEightDotThreeFileName() (value string, err error) {
	retValue, err := instance.GetProperty("EightDotThreeFileName")
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

// SetEncrypted sets the value of Encrypted for the instance
func (instance *CIM_LogicalFile) SetPropertyEncrypted(value bool) (err error) {
	return instance.SetProperty("Encrypted", (value))
}

// GetEncrypted gets the value of Encrypted for the instance
func (instance *CIM_LogicalFile) GetPropertyEncrypted() (value bool, err error) {
	retValue, err := instance.GetProperty("Encrypted")
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

// SetEncryptionMethod sets the value of EncryptionMethod for the instance
func (instance *CIM_LogicalFile) SetPropertyEncryptionMethod(value string) (err error) {
	return instance.SetProperty("EncryptionMethod", (value))
}

// GetEncryptionMethod gets the value of EncryptionMethod for the instance
func (instance *CIM_LogicalFile) GetPropertyEncryptionMethod() (value string, err error) {
	retValue, err := instance.GetProperty("EncryptionMethod")
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

// SetExtension sets the value of Extension for the instance
func (instance *CIM_LogicalFile) SetPropertyExtension(value string) (err error) {
	return instance.SetProperty("Extension", (value))
}

// GetExtension gets the value of Extension for the instance
func (instance *CIM_LogicalFile) GetPropertyExtension() (value string, err error) {
	retValue, err := instance.GetProperty("Extension")
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

// SetFileName sets the value of FileName for the instance
func (instance *CIM_LogicalFile) SetPropertyFileName(value string) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *CIM_LogicalFile) GetPropertyFileName() (value string, err error) {
	retValue, err := instance.GetProperty("FileName")
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
func (instance *CIM_LogicalFile) SetPropertyFileSize(value uint64) (err error) {
	return instance.SetProperty("FileSize", (value))
}

// GetFileSize gets the value of FileSize for the instance
func (instance *CIM_LogicalFile) GetPropertyFileSize() (value uint64, err error) {
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

// SetFileType sets the value of FileType for the instance
func (instance *CIM_LogicalFile) SetPropertyFileType(value string) (err error) {
	return instance.SetProperty("FileType", (value))
}

// GetFileType gets the value of FileType for the instance
func (instance *CIM_LogicalFile) GetPropertyFileType() (value string, err error) {
	retValue, err := instance.GetProperty("FileType")
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

// SetFSCreationClassName sets the value of FSCreationClassName for the instance
func (instance *CIM_LogicalFile) SetPropertyFSCreationClassName(value string) (err error) {
	return instance.SetProperty("FSCreationClassName", (value))
}

// GetFSCreationClassName gets the value of FSCreationClassName for the instance
func (instance *CIM_LogicalFile) GetPropertyFSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("FSCreationClassName")
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

// SetFSName sets the value of FSName for the instance
func (instance *CIM_LogicalFile) SetPropertyFSName(value string) (err error) {
	return instance.SetProperty("FSName", (value))
}

// GetFSName gets the value of FSName for the instance
func (instance *CIM_LogicalFile) GetPropertyFSName() (value string, err error) {
	retValue, err := instance.GetProperty("FSName")
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

// SetHidden sets the value of Hidden for the instance
func (instance *CIM_LogicalFile) SetPropertyHidden(value bool) (err error) {
	return instance.SetProperty("Hidden", (value))
}

// GetHidden gets the value of Hidden for the instance
func (instance *CIM_LogicalFile) GetPropertyHidden() (value bool, err error) {
	retValue, err := instance.GetProperty("Hidden")
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

// SetInUseCount sets the value of InUseCount for the instance
func (instance *CIM_LogicalFile) SetPropertyInUseCount(value uint64) (err error) {
	return instance.SetProperty("InUseCount", (value))
}

// GetInUseCount gets the value of InUseCount for the instance
func (instance *CIM_LogicalFile) GetPropertyInUseCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("InUseCount")
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

// SetLastAccessed sets the value of LastAccessed for the instance
func (instance *CIM_LogicalFile) SetPropertyLastAccessed(value string) (err error) {
	return instance.SetProperty("LastAccessed", (value))
}

// GetLastAccessed gets the value of LastAccessed for the instance
func (instance *CIM_LogicalFile) GetPropertyLastAccessed() (value string, err error) {
	retValue, err := instance.GetProperty("LastAccessed")
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

// SetLastModified sets the value of LastModified for the instance
func (instance *CIM_LogicalFile) SetPropertyLastModified(value string) (err error) {
	return instance.SetProperty("LastModified", (value))
}

// GetLastModified gets the value of LastModified for the instance
func (instance *CIM_LogicalFile) GetPropertyLastModified() (value string, err error) {
	retValue, err := instance.GetProperty("LastModified")
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

// SetPath sets the value of Path for the instance
func (instance *CIM_LogicalFile) SetPropertyPath(value string) (err error) {
	return instance.SetProperty("Path", (value))
}

// GetPath gets the value of Path for the instance
func (instance *CIM_LogicalFile) GetPropertyPath() (value string, err error) {
	retValue, err := instance.GetProperty("Path")
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

// SetReadable sets the value of Readable for the instance
func (instance *CIM_LogicalFile) SetPropertyReadable(value bool) (err error) {
	return instance.SetProperty("Readable", (value))
}

// GetReadable gets the value of Readable for the instance
func (instance *CIM_LogicalFile) GetPropertyReadable() (value bool, err error) {
	retValue, err := instance.GetProperty("Readable")
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

// SetSystem sets the value of System for the instance
func (instance *CIM_LogicalFile) SetPropertySystem(value bool) (err error) {
	return instance.SetProperty("System", (value))
}

// GetSystem gets the value of System for the instance
func (instance *CIM_LogicalFile) GetPropertySystem() (value bool, err error) {
	retValue, err := instance.GetProperty("System")
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

// SetWriteable sets the value of Writeable for the instance
func (instance *CIM_LogicalFile) SetPropertyWriteable(value bool) (err error) {
	return instance.SetProperty("Writeable", (value))
}

// GetWriteable gets the value of Writeable for the instance
func (instance *CIM_LogicalFile) GetPropertyWriteable() (value bool, err error) {
	retValue, err := instance.GetProperty("Writeable")
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

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) TakeOwnerShip() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("TakeOwnerShip")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Option" type="uint32 "></param>
// <param name="SecurityDescriptor" type="Win32_SecurityDescriptor "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) ChangeSecurityPermissions( /* IN */ SecurityDescriptor Win32_SecurityDescriptor,
	/* IN */ Option uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ChangeSecurityPermissions", SecurityDescriptor, Option)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="FileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) Copy( /* IN */ FileName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Copy", FileName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="FileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) Rename( /* IN */ FileName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Rename", FileName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) Delete() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Delete")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) Compress() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Compress")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalFile) Uncompress() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Uncompress")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Recursive" type="bool "></param>
// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) TakeOwnerShipEx( /* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string,
	/* OPTIONAL IN */ Recursive bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("TakeOwnerShipEx", StartFileName, Recursive)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Option" type="uint32 "></param>
// <param name="Recursive" type="bool "></param>
// <param name="SecurityDescriptor" type="Win32_SecurityDescriptor "></param>
// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) ChangeSecurityPermissionsEx( /* IN */ SecurityDescriptor Win32_SecurityDescriptor,
	/* IN */ Option uint32,
	/* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string,
	/* OPTIONAL IN */ Recursive bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("ChangeSecurityPermissionsEx", SecurityDescriptor, Option, StartFileName, Recursive)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FileName" type="string "></param>
// <param name="Recursive" type="bool "></param>
// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) CopyEx( /* IN */ FileName string,
	/* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string,
	/* OPTIONAL IN */ Recursive bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CopyEx", FileName, StartFileName, Recursive)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) DeleteEx( /* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteEx", StartFileName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Recursive" type="bool "></param>
// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) CompressEx( /* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string,
	/* OPTIONAL IN */ Recursive bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CompressEx", StartFileName, Recursive)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Recursive" type="bool "></param>
// <param name="StartFileName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StopFileName" type="string "></param>
func (instance *CIM_LogicalFile) UncompressEx( /* OUT */ StopFileName string,
	/* OPTIONAL IN */ StartFileName string,
	/* OPTIONAL IN */ Recursive bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("UncompressEx", StartFileName, Recursive)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Permissions" type="uint32 "></param>

// <param name="ReturnValue" type="bool "></param>
func (instance *CIM_LogicalFile) GetEffectivePermission( /* IN */ Permissions uint32) (result bool, err error) {
	retVal, err := instance.InvokeMethodWithReturn("GetEffectivePermission", Permissions)
	if err != nil {
		return
	}
	result = (retVal > 0)
	return

}
