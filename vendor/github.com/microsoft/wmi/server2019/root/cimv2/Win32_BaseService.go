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

// Win32_BaseService struct
type Win32_BaseService struct {
	*CIM_Service

	//
	AcceptPause bool

	//
	AcceptStop bool

	//
	DesktopInteract bool

	//
	DisplayName string

	//
	ErrorControl string

	//
	ExitCode uint32

	//
	PathName string

	//
	ServiceSpecificExitCode uint32

	//
	ServiceType string

	//
	StartName string

	//
	State string

	//
	TagId uint32
}

func NewWin32_BaseServiceEx1(instance *cim.WmiInstance) (newInstance *Win32_BaseService, err error) {
	tmp, err := NewCIM_ServiceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_BaseService{
		CIM_Service: tmp,
	}
	return
}

func NewWin32_BaseServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_BaseService, err error) {
	tmp, err := NewCIM_ServiceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_BaseService{
		CIM_Service: tmp,
	}
	return
}

// SetAcceptPause sets the value of AcceptPause for the instance
func (instance *Win32_BaseService) SetPropertyAcceptPause(value bool) (err error) {
	return instance.SetProperty("AcceptPause", (value))
}

// GetAcceptPause gets the value of AcceptPause for the instance
func (instance *Win32_BaseService) GetPropertyAcceptPause() (value bool, err error) {
	retValue, err := instance.GetProperty("AcceptPause")
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

// SetAcceptStop sets the value of AcceptStop for the instance
func (instance *Win32_BaseService) SetPropertyAcceptStop(value bool) (err error) {
	return instance.SetProperty("AcceptStop", (value))
}

// GetAcceptStop gets the value of AcceptStop for the instance
func (instance *Win32_BaseService) GetPropertyAcceptStop() (value bool, err error) {
	retValue, err := instance.GetProperty("AcceptStop")
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

// SetDesktopInteract sets the value of DesktopInteract for the instance
func (instance *Win32_BaseService) SetPropertyDesktopInteract(value bool) (err error) {
	return instance.SetProperty("DesktopInteract", (value))
}

// GetDesktopInteract gets the value of DesktopInteract for the instance
func (instance *Win32_BaseService) GetPropertyDesktopInteract() (value bool, err error) {
	retValue, err := instance.GetProperty("DesktopInteract")
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

// SetDisplayName sets the value of DisplayName for the instance
func (instance *Win32_BaseService) SetPropertyDisplayName(value string) (err error) {
	return instance.SetProperty("DisplayName", (value))
}

// GetDisplayName gets the value of DisplayName for the instance
func (instance *Win32_BaseService) GetPropertyDisplayName() (value string, err error) {
	retValue, err := instance.GetProperty("DisplayName")
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

// SetErrorControl sets the value of ErrorControl for the instance
func (instance *Win32_BaseService) SetPropertyErrorControl(value string) (err error) {
	return instance.SetProperty("ErrorControl", (value))
}

// GetErrorControl gets the value of ErrorControl for the instance
func (instance *Win32_BaseService) GetPropertyErrorControl() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorControl")
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

// SetExitCode sets the value of ExitCode for the instance
func (instance *Win32_BaseService) SetPropertyExitCode(value uint32) (err error) {
	return instance.SetProperty("ExitCode", (value))
}

// GetExitCode gets the value of ExitCode for the instance
func (instance *Win32_BaseService) GetPropertyExitCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExitCode")
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

// SetPathName sets the value of PathName for the instance
func (instance *Win32_BaseService) SetPropertyPathName(value string) (err error) {
	return instance.SetProperty("PathName", (value))
}

// GetPathName gets the value of PathName for the instance
func (instance *Win32_BaseService) GetPropertyPathName() (value string, err error) {
	retValue, err := instance.GetProperty("PathName")
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

// SetServiceSpecificExitCode sets the value of ServiceSpecificExitCode for the instance
func (instance *Win32_BaseService) SetPropertyServiceSpecificExitCode(value uint32) (err error) {
	return instance.SetProperty("ServiceSpecificExitCode", (value))
}

// GetServiceSpecificExitCode gets the value of ServiceSpecificExitCode for the instance
func (instance *Win32_BaseService) GetPropertyServiceSpecificExitCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServiceSpecificExitCode")
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

// SetServiceType sets the value of ServiceType for the instance
func (instance *Win32_BaseService) SetPropertyServiceType(value string) (err error) {
	return instance.SetProperty("ServiceType", (value))
}

// GetServiceType gets the value of ServiceType for the instance
func (instance *Win32_BaseService) GetPropertyServiceType() (value string, err error) {
	retValue, err := instance.GetProperty("ServiceType")
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

// SetStartName sets the value of StartName for the instance
func (instance *Win32_BaseService) SetPropertyStartName(value string) (err error) {
	return instance.SetProperty("StartName", (value))
}

// GetStartName gets the value of StartName for the instance
func (instance *Win32_BaseService) GetPropertyStartName() (value string, err error) {
	retValue, err := instance.GetProperty("StartName")
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

// SetState sets the value of State for the instance
func (instance *Win32_BaseService) SetPropertyState(value string) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *Win32_BaseService) GetPropertyState() (value string, err error) {
	retValue, err := instance.GetProperty("State")
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

// SetTagId sets the value of TagId for the instance
func (instance *Win32_BaseService) SetPropertyTagId(value uint32) (err error) {
	return instance.SetProperty("TagId", (value))
}

// GetTagId gets the value of TagId for the instance
func (instance *Win32_BaseService) GetPropertyTagId() (value uint32, err error) {
	retValue, err := instance.GetProperty("TagId")
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

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) PauseService() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("PauseService")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) ResumeService() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ResumeService")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) InterrogateService() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("InterrogateService")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ControlCode" type="uint8 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) UserControlService( /* IN */ ControlCode uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("UserControlService", ControlCode)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DesktopInteract" type="bool "></param>
// <param name="DisplayName" type="string "></param>
// <param name="ErrorControl" type="uint8 "></param>
// <param name="LoadOrderGroup" type="string "></param>
// <param name="LoadOrderGroupDependencies" type="string []"></param>
// <param name="Name" type="string "></param>
// <param name="PathName" type="string "></param>
// <param name="ServiceDependencies" type="string []"></param>
// <param name="ServiceType" type="uint8 "></param>
// <param name="StartMode" type="string "></param>
// <param name="StartName" type="string "></param>
// <param name="StartPassword" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) Create( /* IN */ Name string,
	/* IN */ DisplayName string,
	/* IN */ PathName string,
	/* IN */ ServiceType uint8,
	/* IN */ ErrorControl uint8,
	/* IN */ StartMode string,
	/* IN */ DesktopInteract bool,
	/* IN */ StartName string,
	/* IN */ StartPassword string,
	/* IN */ LoadOrderGroup string,
	/* IN */ LoadOrderGroupDependencies []string,
	/* IN */ ServiceDependencies []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Create", Name, DisplayName, PathName, ServiceType, ErrorControl, StartMode, DesktopInteract, StartName, StartPassword, LoadOrderGroup, LoadOrderGroupDependencies, ServiceDependencies)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DesktopInteract" type="bool "></param>
// <param name="DisplayName" type="string "></param>
// <param name="ErrorControl" type="uint8 "></param>
// <param name="LoadOrderGroup" type="string "></param>
// <param name="LoadOrderGroupDependencies" type="string []"></param>
// <param name="PathName" type="string "></param>
// <param name="ServiceDependencies" type="string []"></param>
// <param name="ServiceType" type="uint8 "></param>
// <param name="StartMode" type="string "></param>
// <param name="StartName" type="string "></param>
// <param name="StartPassword" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) Change( /* IN */ DisplayName string,
	/* IN */ PathName string,
	/* IN */ ServiceType uint8,
	/* IN */ ErrorControl uint8,
	/* IN */ StartMode string,
	/* IN */ DesktopInteract bool,
	/* IN */ StartName string,
	/* IN */ StartPassword string,
	/* IN */ LoadOrderGroup string,
	/* IN */ LoadOrderGroupDependencies []string,
	/* IN */ ServiceDependencies []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Change", DisplayName, PathName, ServiceType, ErrorControl, StartMode, DesktopInteract, StartName, StartPassword, LoadOrderGroup, LoadOrderGroupDependencies, ServiceDependencies)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="StartMode" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) ChangeStartMode( /* IN */ StartMode string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ChangeStartMode", StartMode)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_BaseService) Delete() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Delete")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
