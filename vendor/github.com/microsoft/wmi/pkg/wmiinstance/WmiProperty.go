// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implements the swbemproperty class
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemproperty

package cim

import (
	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"

	"github.com/microsoft/wmi/go/wmi"
)

type WmiProperty struct {
	session     *WmiSession
	property    *ole.IDispatch
	propertyVar *ole.VARIANT
}

func CreateWmiProperty(propertyVariant *ole.VARIANT, session *WmiSession) (*WmiProperty, error) {
	return &WmiProperty{
		propertyVar: propertyVariant,
		property:    propertyVariant.ToIDispatch(),
		session:     session,
	}, nil
}

// Name
func (c *WmiProperty) Name() string {
	name, err := oleutil.GetProperty(c.property, "Name")
	if err != nil {
		panic("Error retrieving the property Name")
	}

	value, err := GetVariantValue(name)
	if err != nil {
		panic("Error retrieving the property Name")
	}

	return value.(string)
}

// Value
func (c *WmiProperty) Value() interface{} {
	rawSystemProperty, err := oleutil.GetProperty(c.property, "Value")
	if err != nil {
		panic("Error retrieving the property value")
	}

	value, err := GetVariantValue(rawSystemProperty)
	if err != nil {
		panic("Error retrieving the property value")
	}

	return value
}

// Type
func (c *WmiProperty) Type() wmi.WmiType {
	rawSystemProperty, err := oleutil.GetProperty(c.property, "CIMType")
	if err != nil {
		panic("Error retrieving the property type")
	}

	value, err := GetVariantValue(rawSystemProperty)
	if err != nil {
		panic("Error retrieving the property type")
	}

	return wmi.WmiType(value.(int))
}

func CloseAllProperties(properties []*WmiProperty) {
	for _, property := range properties {
		property.Close()
	}
}

// Dispose
func (c *WmiProperty) Close() error {
	return c.propertyVar.Clear()
}
