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

// Win32_PortConnector struct
type Win32_PortConnector struct {
	*CIM_PhysicalConnector

	//
	ExternalReferenceDesignator string

	//
	InternalReferenceDesignator string

	//
	PortType uint16
}

func NewWin32_PortConnectorEx1(instance *cim.WmiInstance) (newInstance *Win32_PortConnector, err error) {
	tmp, err := NewCIM_PhysicalConnectorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PortConnector{
		CIM_PhysicalConnector: tmp,
	}
	return
}

func NewWin32_PortConnectorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PortConnector, err error) {
	tmp, err := NewCIM_PhysicalConnectorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PortConnector{
		CIM_PhysicalConnector: tmp,
	}
	return
}

// SetExternalReferenceDesignator sets the value of ExternalReferenceDesignator for the instance
func (instance *Win32_PortConnector) SetPropertyExternalReferenceDesignator(value string) (err error) {
	return instance.SetProperty("ExternalReferenceDesignator", (value))
}

// GetExternalReferenceDesignator gets the value of ExternalReferenceDesignator for the instance
func (instance *Win32_PortConnector) GetPropertyExternalReferenceDesignator() (value string, err error) {
	retValue, err := instance.GetProperty("ExternalReferenceDesignator")
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

// SetInternalReferenceDesignator sets the value of InternalReferenceDesignator for the instance
func (instance *Win32_PortConnector) SetPropertyInternalReferenceDesignator(value string) (err error) {
	return instance.SetProperty("InternalReferenceDesignator", (value))
}

// GetInternalReferenceDesignator gets the value of InternalReferenceDesignator for the instance
func (instance *Win32_PortConnector) GetPropertyInternalReferenceDesignator() (value string, err error) {
	retValue, err := instance.GetProperty("InternalReferenceDesignator")
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

// SetPortType sets the value of PortType for the instance
func (instance *Win32_PortConnector) SetPropertyPortType(value uint16) (err error) {
	return instance.SetProperty("PortType", (value))
}

// GetPortType gets the value of PortType for the instance
func (instance *Win32_PortConnector) GetPropertyPortType() (value uint16, err error) {
	retValue, err := instance.GetProperty("PortType")
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
