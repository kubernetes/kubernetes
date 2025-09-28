// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implement a wrapper of the SWbemObject class (from an instance perspective).
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemobject

package cim

import (
	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

type WmiClass struct {
	session  *WmiSession
	class    *ole.IDispatch
	classVar *ole.VARIANT
}

// WmiInstanceCollection is a slice of WmiClass
type WmiClassCollection []*WmiClass

// Close all class in a collection
func (wmic *WmiClassCollection) Close() {
	for _, i := range *wmic {
		i.Close()
	}
}

func CreateWmiClass(classVar *ole.VARIANT, session *WmiSession) (*WmiClass, error) {
	return &WmiClass{
		classVar: classVar,
		class:    classVar.ToIDispatch(),
		session:  session,
	}, nil
}

// Makes a new instance of the class
func (c *WmiClass) MakeInstance() (*WmiInstance, error) {
	rawResult, err := oleutil.CallMethod(c.class, "SpawnInstance_")
	if err != nil {
		return nil, err
	}

	return CreateWmiInstance(rawResult, c.session)
}

func (c *WmiClass) mustGetSystemProperty(name string) *WmiProperty {
	wmiProperty, err := c.GetSystemProperty(name)
	if err != nil {
		panic("Couldn't retreive a system property. GetSystemProperty failed")
	}

	return wmiProperty
}

func (c *WmiClass) GetSystemProperty(name string) (*WmiProperty, error) {
	// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemobjectex-systemproperties-
	rawResult, err := oleutil.GetProperty(c.class, "SystemProperties_")
	if err != nil {
		return nil, err
	}

	// SWbemObjectEx.SystemProperties_ returns
	// an SWbemPropertySet object that contains the collection
	// of sytem properties for the c class
	sWbemObjectExAsIDispatch := rawResult.ToIDispatch()
	defer rawResult.Clear()

	// Get the system property
	sWbemProperty, err := oleutil.CallMethod(sWbemObjectExAsIDispatch, "Item", name)
	if err != nil {
		return nil, err
	}

	return CreateWmiProperty(sWbemProperty, c.session)
}

// ClassName
func (c *WmiClass) GetClassName() string {
	class := c.mustGetSystemProperty("__CLASS")
	defer class.Close()

	return class.Value().(string)
}

// SetClassName
func (c *WmiClass) SetClassName(name string) error {
	rawResult, err := oleutil.GetProperty(c.class, "Path_")
	if err != nil {
		return err
	}

	pathIDispatch := rawResult.ToIDispatch()
	defer rawResult.Clear()

	classRawResult, err := oleutil.PutProperty(pathIDispatch, "Class", name)
	if err != nil {
		return err
	}
	defer classRawResult.Clear()

	return nil
}

// SuperClassName
func (c *WmiClass) GetSuperClassName() string {
	superclass := c.mustGetSystemProperty("__SUPERCLASS")
	defer superclass.Close()

	return superclass.Value().(string)
}

// ServerName
func (c *WmiClass) GetServerName() string {
	server := c.mustGetSystemProperty("__SERVER")
	defer server.Close()

	return server.Value().(string)
}

// Namespace
func (c *WmiClass) GetNamespace() string {
	namespace := c.mustGetSystemProperty("__NAMESPACE")
	defer namespace.Close()

	return namespace.Value().(string)
}

// SuperClass
func (c *WmiClass) GetSuperClass() *WmiClass {
	class, err := c.session.GetClass(c.GetSuperClassName())
	if err != nil {
		panic("The class for this instance doesn't exist")
	}

	return class
}

// Derivation
func (c *WmiClass) GetDerivation() []string {
	valueNameProperty, err := oleutil.GetProperty(c.class, "Derivation_")
	if err != nil {
		panic("GetDerivation() failed to get the Derivation_ name property")
	}
	defer valueNameProperty.Clear()

	derivations, err := GetVariantValues(valueNameProperty)
	if len(derivations) < 1 {
		panic("GetDerivation() failed to get the Derivation_ values")
	}

	values := []string{}
	for _, derivation := range derivations {
		values = append(values, derivation.(string))
	}

	return values
}

// Properties
func (c *WmiClass) GetPropertiesNames() []string {
	values := c.getValueList("Properties_")

	valueNames := []string{}
	for _, value := range values {
		valueNames = append(valueNames, value.Name())
	}
	CloseAllProperties(values)

	return valueNames
}

// Qualifiers
func (c *WmiClass) GetQualifiersNames() []string {
	values := c.getValueList("Qualifiers_")

	valueNames := []string{}
	for _, value := range values {
		valueNames = append(valueNames, value.Name())
	}
	CloseAllProperties(values)

	return valueNames
}

// Methods
func (c *WmiClass) GetMethodsNames() []string {
	values := c.getValueList("Methods_")

	valueNames := []string{}
	for _, value := range values {
		valueNames = append(valueNames, value.Name())
	}
	CloseAllProperties(values)

	return valueNames
}

// GetProperty gets the property of the instance specified by name and returns in value
func (c *WmiClass) GetProperty(name string) (interface{}, error) {
	rawResult, err := oleutil.GetProperty(c.class, name)
	if err != nil {
		return nil, err
	}

	defer rawResult.Clear()

	if rawResult.VT == 0x1 {
		return nil, err
	}

	return GetVariantValue(rawResult)
}

// SetProperty sets a value of property representation by name with value
func (c *WmiClass) SetProperty(name string, value interface{}) error {
	rawResult, err := oleutil.PutProperty(c.class, name, value)
	if err != nil {
		return err
	}

	defer rawResult.Clear()
	return nil
}

// Commit
func (c *WmiClass) Commit() error {
	rawResult, err := oleutil.CallMethod(c.class, "Put_")
	if err != nil {
		return err
	}
	defer rawResult.Clear()
	return nil

}

// Modify
func (c *WmiClass) Modify() error {
	return c.Commit()
}

func (c *WmiClass) getValueList(valuePropertyName string) []*WmiProperty {
	valuesProperty, err := oleutil.GetProperty(c.class, valuePropertyName)
	if err != nil {
		panic("getValueList failed getting valuePropertyName")
	}
	defer valuesProperty.Clear()

	result := valuesProperty.ToIDispatch()
	// Doc: https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/dispid-constants
	enum_property, err := result.GetProperty("_NewEnum")
	if err != nil {
		panic("getValueList() failed getting _NewEnum")
	}
	defer enum_property.Clear()

	// https://docs.microsoft.com/en-us/windows/win32/api/oaidl/nn-oaidl-ienumvariant
	enum, err := enum_property.ToIUnknown().IEnumVARIANT(ole.IID_IEnumVariant)
	if err != nil {
		panic("getValueList() failed getting IID_IEnumVariant")
	}
	if enum == nil {
		return []*WmiProperty{}
	}
	defer enum.Release()

	properties := []*WmiProperty{}
	for valueVariant, length, err := enum.Next(1); length > 0; valueVariant, length, err = enum.Next(1) {
		if err != nil {
			panic("getValueList() failed to browse the value list")
		}

		property, err := CreateWmiProperty(&valueVariant, c.session)
		if err != nil {
			panic("getValueList() failed to create the WMI property")
		}

		properties = append(properties, property)
	}

	return properties
}

// MethodParameters
func (c *WmiClass) MethodParameters(methodName string) []string {
	panic("not implemented")
	// TODO. Relevant docs:
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemmethodset
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemmethod
}

// Invoke static method on a wmi class
func (c *WmiClass) InvokeMethod(methodName string, params ...interface{}) ([]interface{}, error) {
	rawResult, err := oleutil.CallMethod(c.class, methodName, params...)
	if err != nil {
		return nil, err
	}
	defer rawResult.Clear()
	values, err := GetVariantValues(rawResult)
	return values, err
}

// CloseAllClasses
func CloseAllClasses(classes []*WmiClass) {
	for _, class := range classes {
		class.Close()
	}
}

// Dispose
func (c *WmiClass) Close() error {
	return c.classVar.Clear()
}
