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

// Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration struct
type Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration struct {
	*Win32_PerfRawData

	//
	Cacheupdateaveragemilliseconds uint32

	//
	Cacheupdatecount uint32

	//
	Commitaveragemilliseconds uint32

	//
	Commitbytespersecondaverage uint32

	//
	Commitcount uint32

	//
	Compactaveragemilliseconds uint32

	//
	Compactcount uint32

	//
	Configlockacquireaveragemilliseconds uint32

	//
	Configlockcount uint32

	//
	Filelockacquireaveragemilliseconds uint32

	//
	Filelockreleaseaveragemilliseconds uint32

	//
	Getaveragemilliseconds uint32

	//
	Getcount uint32

	//
	Loadfileaveragemilliseconds uint32

	//
	Lockacquireaveragemilliseconds uint32

	//
	Lockcount uint32

	//
	Lockreleaseaveragemilliseconds uint32

	//
	Querysizeaveragesizemilliseconds uint32

	//
	Querysizecount uint32

	//
	Readbytes uint32

	//
	Readbytespersecondaverage uint32

	//
	Readfilebytes uint32

	//
	Readfilebytespersecondaverage uint32

	//
	Readfilecount uint32

	//
	Removeaveragemilliseconds uint32

	//
	Removecount uint32

	//
	Setaveragemilliseconds uint32

	//
	Setcount uint32

	//
	Writebytes uint32

	//
	Writebytespersecondaverage uint32

	//
	Writefilebytes uint32

	//
	Writefilecount uint32
}

func NewWin32_PerfRawData_HyperVStorageStats_HyperVConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_HyperVStorageStats_HyperVConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCacheupdateaveragemilliseconds sets the value of Cacheupdateaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCacheupdateaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Cacheupdateaveragemilliseconds", (value))
}

// GetCacheupdateaveragemilliseconds gets the value of Cacheupdateaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCacheupdateaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Cacheupdateaveragemilliseconds")
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

// SetCacheupdatecount sets the value of Cacheupdatecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCacheupdatecount(value uint32) (err error) {
	return instance.SetProperty("Cacheupdatecount", (value))
}

// GetCacheupdatecount gets the value of Cacheupdatecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCacheupdatecount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Cacheupdatecount")
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

// SetCommitaveragemilliseconds sets the value of Commitaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCommitaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Commitaveragemilliseconds", (value))
}

// GetCommitaveragemilliseconds gets the value of Commitaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCommitaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Commitaveragemilliseconds")
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

// SetCommitbytespersecondaverage sets the value of Commitbytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCommitbytespersecondaverage(value uint32) (err error) {
	return instance.SetProperty("Commitbytespersecondaverage", (value))
}

// GetCommitbytespersecondaverage gets the value of Commitbytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCommitbytespersecondaverage() (value uint32, err error) {
	retValue, err := instance.GetProperty("Commitbytespersecondaverage")
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

// SetCommitcount sets the value of Commitcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCommitcount(value uint32) (err error) {
	return instance.SetProperty("Commitcount", (value))
}

// GetCommitcount gets the value of Commitcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCommitcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Commitcount")
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

// SetCompactaveragemilliseconds sets the value of Compactaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCompactaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Compactaveragemilliseconds", (value))
}

// GetCompactaveragemilliseconds gets the value of Compactaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCompactaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Compactaveragemilliseconds")
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

// SetCompactcount sets the value of Compactcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyCompactcount(value uint32) (err error) {
	return instance.SetProperty("Compactcount", (value))
}

// GetCompactcount gets the value of Compactcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyCompactcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Compactcount")
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

// SetConfiglockacquireaveragemilliseconds sets the value of Configlockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyConfiglockacquireaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Configlockacquireaveragemilliseconds", (value))
}

// GetConfiglockacquireaveragemilliseconds gets the value of Configlockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyConfiglockacquireaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Configlockacquireaveragemilliseconds")
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

// SetConfiglockcount sets the value of Configlockcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyConfiglockcount(value uint32) (err error) {
	return instance.SetProperty("Configlockcount", (value))
}

// GetConfiglockcount gets the value of Configlockcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyConfiglockcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Configlockcount")
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

// SetFilelockacquireaveragemilliseconds sets the value of Filelockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyFilelockacquireaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Filelockacquireaveragemilliseconds", (value))
}

// GetFilelockacquireaveragemilliseconds gets the value of Filelockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyFilelockacquireaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Filelockacquireaveragemilliseconds")
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

// SetFilelockreleaseaveragemilliseconds sets the value of Filelockreleaseaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyFilelockreleaseaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Filelockreleaseaveragemilliseconds", (value))
}

// GetFilelockreleaseaveragemilliseconds gets the value of Filelockreleaseaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyFilelockreleaseaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Filelockreleaseaveragemilliseconds")
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

// SetGetaveragemilliseconds sets the value of Getaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyGetaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Getaveragemilliseconds", (value))
}

// GetGetaveragemilliseconds gets the value of Getaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyGetaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Getaveragemilliseconds")
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

// SetGetcount sets the value of Getcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyGetcount(value uint32) (err error) {
	return instance.SetProperty("Getcount", (value))
}

// GetGetcount gets the value of Getcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyGetcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Getcount")
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

// SetLoadfileaveragemilliseconds sets the value of Loadfileaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyLoadfileaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Loadfileaveragemilliseconds", (value))
}

// GetLoadfileaveragemilliseconds gets the value of Loadfileaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyLoadfileaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Loadfileaveragemilliseconds")
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

// SetLockacquireaveragemilliseconds sets the value of Lockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyLockacquireaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Lockacquireaveragemilliseconds", (value))
}

// GetLockacquireaveragemilliseconds gets the value of Lockacquireaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyLockacquireaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Lockacquireaveragemilliseconds")
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

// SetLockcount sets the value of Lockcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyLockcount(value uint32) (err error) {
	return instance.SetProperty("Lockcount", (value))
}

// GetLockcount gets the value of Lockcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyLockcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Lockcount")
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

// SetLockreleaseaveragemilliseconds sets the value of Lockreleaseaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyLockreleaseaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Lockreleaseaveragemilliseconds", (value))
}

// GetLockreleaseaveragemilliseconds gets the value of Lockreleaseaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyLockreleaseaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Lockreleaseaveragemilliseconds")
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

// SetQuerysizeaveragesizemilliseconds sets the value of Querysizeaveragesizemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyQuerysizeaveragesizemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Querysizeaveragesizemilliseconds", (value))
}

// GetQuerysizeaveragesizemilliseconds gets the value of Querysizeaveragesizemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyQuerysizeaveragesizemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Querysizeaveragesizemilliseconds")
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

// SetQuerysizecount sets the value of Querysizecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyQuerysizecount(value uint32) (err error) {
	return instance.SetProperty("Querysizecount", (value))
}

// GetQuerysizecount gets the value of Querysizecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyQuerysizecount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Querysizecount")
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

// SetReadbytes sets the value of Readbytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyReadbytes(value uint32) (err error) {
	return instance.SetProperty("Readbytes", (value))
}

// GetReadbytes gets the value of Readbytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyReadbytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Readbytes")
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

// SetReadbytespersecondaverage sets the value of Readbytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyReadbytespersecondaverage(value uint32) (err error) {
	return instance.SetProperty("Readbytespersecondaverage", (value))
}

// GetReadbytespersecondaverage gets the value of Readbytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyReadbytespersecondaverage() (value uint32, err error) {
	retValue, err := instance.GetProperty("Readbytespersecondaverage")
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

// SetReadfilebytes sets the value of Readfilebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyReadfilebytes(value uint32) (err error) {
	return instance.SetProperty("Readfilebytes", (value))
}

// GetReadfilebytes gets the value of Readfilebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyReadfilebytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Readfilebytes")
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

// SetReadfilebytespersecondaverage sets the value of Readfilebytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyReadfilebytespersecondaverage(value uint32) (err error) {
	return instance.SetProperty("Readfilebytespersecondaverage", (value))
}

// GetReadfilebytespersecondaverage gets the value of Readfilebytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyReadfilebytespersecondaverage() (value uint32, err error) {
	retValue, err := instance.GetProperty("Readfilebytespersecondaverage")
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

// SetReadfilecount sets the value of Readfilecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyReadfilecount(value uint32) (err error) {
	return instance.SetProperty("Readfilecount", (value))
}

// GetReadfilecount gets the value of Readfilecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyReadfilecount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Readfilecount")
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

// SetRemoveaveragemilliseconds sets the value of Removeaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyRemoveaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Removeaveragemilliseconds", (value))
}

// GetRemoveaveragemilliseconds gets the value of Removeaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyRemoveaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Removeaveragemilliseconds")
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

// SetRemovecount sets the value of Removecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyRemovecount(value uint32) (err error) {
	return instance.SetProperty("Removecount", (value))
}

// GetRemovecount gets the value of Removecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyRemovecount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Removecount")
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

// SetSetaveragemilliseconds sets the value of Setaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertySetaveragemilliseconds(value uint32) (err error) {
	return instance.SetProperty("Setaveragemilliseconds", (value))
}

// GetSetaveragemilliseconds gets the value of Setaveragemilliseconds for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertySetaveragemilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Setaveragemilliseconds")
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

// SetSetcount sets the value of Setcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertySetcount(value uint32) (err error) {
	return instance.SetProperty("Setcount", (value))
}

// GetSetcount gets the value of Setcount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertySetcount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Setcount")
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

// SetWritebytes sets the value of Writebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyWritebytes(value uint32) (err error) {
	return instance.SetProperty("Writebytes", (value))
}

// GetWritebytes gets the value of Writebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyWritebytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Writebytes")
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

// SetWritebytespersecondaverage sets the value of Writebytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyWritebytespersecondaverage(value uint32) (err error) {
	return instance.SetProperty("Writebytespersecondaverage", (value))
}

// GetWritebytespersecondaverage gets the value of Writebytespersecondaverage for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyWritebytespersecondaverage() (value uint32, err error) {
	retValue, err := instance.GetProperty("Writebytespersecondaverage")
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

// SetWritefilebytes sets the value of Writefilebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyWritefilebytes(value uint32) (err error) {
	return instance.SetProperty("Writefilebytes", (value))
}

// GetWritefilebytes gets the value of Writefilebytes for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyWritefilebytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Writefilebytes")
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

// SetWritefilecount sets the value of Writefilecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) SetPropertyWritefilecount(value uint32) (err error) {
	return instance.SetProperty("Writefilecount", (value))
}

// GetWritefilecount gets the value of Writefilecount for the instance
func (instance *Win32_PerfRawData_HyperVStorageStats_HyperVConfiguration) GetPropertyWritefilecount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Writefilecount")
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
