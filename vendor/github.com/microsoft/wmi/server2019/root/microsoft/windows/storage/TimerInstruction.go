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

// __TimerInstruction struct
type __TimerInstruction struct {
	*__EventGenerator

	//
	SkipIfPassed bool

	//
	TimerId string
}

func New__TimerInstructionEx1(instance *cim.WmiInstance) (newInstance *__TimerInstruction, err error) {
	tmp, err := New__EventGeneratorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__TimerInstruction{
		__EventGenerator: tmp,
	}
	return
}

func New__TimerInstructionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__TimerInstruction, err error) {
	tmp, err := New__EventGeneratorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__TimerInstruction{
		__EventGenerator: tmp,
	}
	return
}

// SetSkipIfPassed sets the value of SkipIfPassed for the instance
func (instance *__TimerInstruction) SetPropertySkipIfPassed(value bool) (err error) {
	return instance.SetProperty("SkipIfPassed", (value))
}

// GetSkipIfPassed gets the value of SkipIfPassed for the instance
func (instance *__TimerInstruction) GetPropertySkipIfPassed() (value bool, err error) {
	retValue, err := instance.GetProperty("SkipIfPassed")
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

// SetTimerId sets the value of TimerId for the instance
func (instance *__TimerInstruction) SetPropertyTimerId(value string) (err error) {
	return instance.SetProperty("TimerId", (value))
}

// GetTimerId gets the value of TimerId for the instance
func (instance *__TimerInstruction) GetPropertyTimerId() (value string, err error) {
	retValue, err := instance.GetProperty("TimerId")
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
