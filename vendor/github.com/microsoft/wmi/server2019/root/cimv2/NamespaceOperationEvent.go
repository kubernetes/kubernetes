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

// __NamespaceOperationEvent struct
type __NamespaceOperationEvent struct {
	*__Event

	//
	TargetNamespace __Namespace
}

func New__NamespaceOperationEventEx1(instance *cim.WmiInstance) (newInstance *__NamespaceOperationEvent, err error) {
	tmp, err := New__EventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__NamespaceOperationEvent{
		__Event: tmp,
	}
	return
}

func New__NamespaceOperationEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__NamespaceOperationEvent, err error) {
	tmp, err := New__EventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__NamespaceOperationEvent{
		__Event: tmp,
	}
	return
}

// SetTargetNamespace sets the value of TargetNamespace for the instance
func (instance *__NamespaceOperationEvent) SetPropertyTargetNamespace(value __Namespace) (err error) {
	return instance.SetProperty("TargetNamespace", (value))
}

// GetTargetNamespace gets the value of TargetNamespace for the instance
func (instance *__NamespaceOperationEvent) GetPropertyTargetNamespace() (value __Namespace, err error) {
	retValue, err := instance.GetProperty("TargetNamespace")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(__Namespace)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " __Namespace is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = __Namespace(valuetmp)

	return
}
