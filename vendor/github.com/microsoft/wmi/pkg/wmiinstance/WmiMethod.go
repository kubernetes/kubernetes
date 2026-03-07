// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implements the swbemproperty class
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemmethod

package cim

import (
	"log"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
	"github.com/microsoft/wmi/pkg/errors"
)

type WmiMethod struct {
	Name string
	// Reference
	session *WmiSession
	// Reference
	classInstance *WmiInstance
}

type WmiMethodResult struct {
	ReturnValue     int32
	OutMethodParams map[string]*WmiMethodParam
}

// NewWmiMethod
func NewWmiMethod(methodName string, instance *WmiInstance) (*WmiMethod, error) {
	return &WmiMethod{
		Name:          methodName,
		classInstance: instance,
		session:       instance.GetSession(),
	}, nil
}

func (c *WmiMethod) addInParam(inparamVariant *ole.VARIANT, paramName string, paramValue interface{}) error {
	rawProperties, err := inparamVariant.ToIDispatch().GetProperty("Properties_")
	if err != nil {
		return err
	}
	defer rawProperties.Clear()
	rawProperty, err := rawProperties.ToIDispatch().CallMethod("Item", paramName)
	if err != nil {
		return err
	}
	defer rawProperty.Clear()

	p, err := rawProperty.ToIDispatch().PutProperty("Value", paramValue)
	if err != nil {
		return err
	}
	defer p.Clear()
	return nil
}

func (c *WmiMethod) Execute(inParam, outParam WmiMethodParamCollection) (result *WmiMethodResult, err error) {
	log.Printf("[WMI] - Executing Method [%s]\n", c.Name)

	iDispatchInstance := c.classInstance.GetIDispatch()
	if iDispatchInstance == nil {
		return nil, errors.Wrapf(errors.InvalidInput, "InvalidInstance")
	}
	rawResult, err := iDispatchInstance.GetProperty("Methods_")
	if err != nil {
		return nil, err
	}
	defer rawResult.Clear()
	// Retrive the method
	rawMethod, err := rawResult.ToIDispatch().CallMethod("Item", c.Name)
	if err != nil {
		return nil, err
	}
	defer rawMethod.Clear()

	params := []interface{}{c.Name}

	if len(inParam) > 0 {
		inparamsRaw, err := rawMethod.ToIDispatch().GetProperty("InParameters")
		if err != nil {
			return nil, err
		}
		defer inparamsRaw.Clear()

		// Method with no parameters may return a VARIANT with nil IDispatch
		if inparamsRaw.Val != 0 {
			inparams, err := oleutil.CallMethod(inparamsRaw.ToIDispatch(), "SpawnInstance_")
			if err != nil {
				return nil, err
			}
			defer inparams.Clear()

			for _, inp := range inParam {
				// 	log.Printf("InParam [%s]=>[%+v]\n", inp.Name, inp.Value)
				c.addInParam(inparams, inp.Name, inp.Value)
			}

			params = append(params, inparams)
		}
	}

	result = &WmiMethodResult{
		OutMethodParams: map[string]*WmiMethodParam{},
	}
	outparams, err := c.classInstance.GetIDispatch().CallMethod("ExecMethod_", params...)
	if err != nil {
		return
	}
	defer outparams.Clear()
	returnRaw, err := outparams.ToIDispatch().GetProperty("ReturnValue")
	if err != nil {
		return
	}
	defer returnRaw.Clear()
	result.ReturnValue = returnRaw.Value().(int32)
	log.Printf("[WMI] - Return [%d] ", result.ReturnValue)

	for _, outp := range outParam {
		returnRawIn, err1 := outparams.ToIDispatch().GetProperty(outp.Name)
		if err1 != nil {
			err = err1
			return
		}
		defer returnRawIn.Clear()

		value, err1 := GetVariantValue(returnRawIn)
		if err1 != nil {
			err = err1
			return
		}
		// log.Printf("OutParam [%s]=> [%+v]\n", outp.Name, value)

		result.OutMethodParams[outp.Name] = NewWmiMethodParam(outp.Name, value)
	}
	return
}

func (c *WmiMethod) Close() error {
	return nil
}

type WmiMethodCollection []*WmiMethod

func (c *WmiMethodCollection) Close() error {
	var err error
	for _, p := range *c {
		err = p.Close()
	}
	return err
}
