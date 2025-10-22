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

// CIM_ExecuteProgram struct
type CIM_ExecuteProgram struct {
	*CIM_Action

	//
	CommandLine string

	//
	ProgramPath string
}

func NewCIM_ExecuteProgramEx1(instance *cim.WmiInstance) (newInstance *CIM_ExecuteProgram, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ExecuteProgram{
		CIM_Action: tmp,
	}
	return
}

func NewCIM_ExecuteProgramEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ExecuteProgram, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ExecuteProgram{
		CIM_Action: tmp,
	}
	return
}

// SetCommandLine sets the value of CommandLine for the instance
func (instance *CIM_ExecuteProgram) SetPropertyCommandLine(value string) (err error) {
	return instance.SetProperty("CommandLine", (value))
}

// GetCommandLine gets the value of CommandLine for the instance
func (instance *CIM_ExecuteProgram) GetPropertyCommandLine() (value string, err error) {
	retValue, err := instance.GetProperty("CommandLine")
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

// SetProgramPath sets the value of ProgramPath for the instance
func (instance *CIM_ExecuteProgram) SetPropertyProgramPath(value string) (err error) {
	return instance.SetProperty("ProgramPath", (value))
}

// GetProgramPath gets the value of ProgramPath for the instance
func (instance *CIM_ExecuteProgram) GetPropertyProgramPath() (value string, err error) {
	retValue, err := instance.GetProperty("ProgramPath")
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
