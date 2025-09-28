// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implements the swbemproperty class
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemproperty

package cim

import (
// "github.com/go-ole/go-ole"
)

type WmiMethodParam struct {
	Name  string
	Value interface{}
	//session     *WmiSession
	//property    *ole.IDispatch
	//propertyVar *ole.VARIANT
}

// func NewWmiMethodParam(name string, val interface{}, propertyVariant *ole.VARIANT, session *WmiSession) (*WmiMethodParam, error) {
func NewWmiMethodParam(name string, val interface{}) *WmiMethodParam {
	return &WmiMethodParam{
		Name: name,
		//propertyVar: propertyVariant,
		//property:    propertyVariant.ToIDispatch(),
		//session: session,
		Value: val,
	}
}

// Dispose
func (c *WmiMethodParam) Close() error {
	return nil //c.propertyVar.Clear()
}

type WmiMethodParamCollection []*WmiMethodParam

func (c *WmiMethodParamCollection) Close() error {
	var err error
	for _, p := range *c {
		err = p.Close()
	}
	return err
}
