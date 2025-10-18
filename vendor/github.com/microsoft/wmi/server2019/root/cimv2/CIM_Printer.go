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

// CIM_Printer struct
type CIM_Printer struct {
	*CIM_LogicalDevice

	//
	AvailableJobSheets []string

	//
	Capabilities []uint16

	//
	CapabilityDescriptions []string

	//
	CharSetsSupported []string

	//
	CurrentCapabilities []uint16

	//
	CurrentCharSet string

	//
	CurrentLanguage uint16

	//
	CurrentMimeType string

	//
	CurrentNaturalLanguage string

	//
	CurrentPaperType string

	//
	DefaultCapabilities []uint16

	//
	DefaultCopies uint32

	//
	DefaultLanguage uint16

	//
	DefaultMimeType string

	//
	DefaultNumberUp uint32

	//
	DefaultPaperType string

	//
	DetectedErrorState uint16

	//
	ErrorInformation []string

	//
	HorizontalResolution uint32

	//
	JobCountSinceLastReset uint32

	//
	LanguagesSupported []uint16

	//
	MarkingTechnology uint16

	//
	MaxCopies uint32

	//
	MaxNumberUp uint32

	//
	MaxSizeSupported uint32

	//
	MimeTypesSupported []string

	//
	NaturalLanguagesSupported []string

	//
	PaperSizesSupported []uint16

	//
	PaperTypesAvailable []string

	//
	PrinterStatus uint16

	//
	TimeOfLastReset string

	//
	VerticalResolution uint32
}

func NewCIM_PrinterEx1(instance *cim.WmiInstance) (newInstance *CIM_Printer, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Printer{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_PrinterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Printer, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Printer{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetAvailableJobSheets sets the value of AvailableJobSheets for the instance
func (instance *CIM_Printer) SetPropertyAvailableJobSheets(value []string) (err error) {
	return instance.SetProperty("AvailableJobSheets", (value))
}

// GetAvailableJobSheets gets the value of AvailableJobSheets for the instance
func (instance *CIM_Printer) GetPropertyAvailableJobSheets() (value []string, err error) {
	retValue, err := instance.GetProperty("AvailableJobSheets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCapabilities sets the value of Capabilities for the instance
func (instance *CIM_Printer) SetPropertyCapabilities(value []uint16) (err error) {
	return instance.SetProperty("Capabilities", (value))
}

// GetCapabilities gets the value of Capabilities for the instance
func (instance *CIM_Printer) GetPropertyCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("Capabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetCapabilityDescriptions sets the value of CapabilityDescriptions for the instance
func (instance *CIM_Printer) SetPropertyCapabilityDescriptions(value []string) (err error) {
	return instance.SetProperty("CapabilityDescriptions", (value))
}

// GetCapabilityDescriptions gets the value of CapabilityDescriptions for the instance
func (instance *CIM_Printer) GetPropertyCapabilityDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("CapabilityDescriptions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCharSetsSupported sets the value of CharSetsSupported for the instance
func (instance *CIM_Printer) SetPropertyCharSetsSupported(value []string) (err error) {
	return instance.SetProperty("CharSetsSupported", (value))
}

// GetCharSetsSupported gets the value of CharSetsSupported for the instance
func (instance *CIM_Printer) GetPropertyCharSetsSupported() (value []string, err error) {
	retValue, err := instance.GetProperty("CharSetsSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCurrentCapabilities sets the value of CurrentCapabilities for the instance
func (instance *CIM_Printer) SetPropertyCurrentCapabilities(value []uint16) (err error) {
	return instance.SetProperty("CurrentCapabilities", (value))
}

// GetCurrentCapabilities gets the value of CurrentCapabilities for the instance
func (instance *CIM_Printer) GetPropertyCurrentCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("CurrentCapabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetCurrentCharSet sets the value of CurrentCharSet for the instance
func (instance *CIM_Printer) SetPropertyCurrentCharSet(value string) (err error) {
	return instance.SetProperty("CurrentCharSet", (value))
}

// GetCurrentCharSet gets the value of CurrentCharSet for the instance
func (instance *CIM_Printer) GetPropertyCurrentCharSet() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentCharSet")
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

// SetCurrentLanguage sets the value of CurrentLanguage for the instance
func (instance *CIM_Printer) SetPropertyCurrentLanguage(value uint16) (err error) {
	return instance.SetProperty("CurrentLanguage", (value))
}

// GetCurrentLanguage gets the value of CurrentLanguage for the instance
func (instance *CIM_Printer) GetPropertyCurrentLanguage() (value uint16, err error) {
	retValue, err := instance.GetProperty("CurrentLanguage")
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

// SetCurrentMimeType sets the value of CurrentMimeType for the instance
func (instance *CIM_Printer) SetPropertyCurrentMimeType(value string) (err error) {
	return instance.SetProperty("CurrentMimeType", (value))
}

// GetCurrentMimeType gets the value of CurrentMimeType for the instance
func (instance *CIM_Printer) GetPropertyCurrentMimeType() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentMimeType")
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

// SetCurrentNaturalLanguage sets the value of CurrentNaturalLanguage for the instance
func (instance *CIM_Printer) SetPropertyCurrentNaturalLanguage(value string) (err error) {
	return instance.SetProperty("CurrentNaturalLanguage", (value))
}

// GetCurrentNaturalLanguage gets the value of CurrentNaturalLanguage for the instance
func (instance *CIM_Printer) GetPropertyCurrentNaturalLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentNaturalLanguage")
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

// SetCurrentPaperType sets the value of CurrentPaperType for the instance
func (instance *CIM_Printer) SetPropertyCurrentPaperType(value string) (err error) {
	return instance.SetProperty("CurrentPaperType", (value))
}

// GetCurrentPaperType gets the value of CurrentPaperType for the instance
func (instance *CIM_Printer) GetPropertyCurrentPaperType() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentPaperType")
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

// SetDefaultCapabilities sets the value of DefaultCapabilities for the instance
func (instance *CIM_Printer) SetPropertyDefaultCapabilities(value []uint16) (err error) {
	return instance.SetProperty("DefaultCapabilities", (value))
}

// GetDefaultCapabilities gets the value of DefaultCapabilities for the instance
func (instance *CIM_Printer) GetPropertyDefaultCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("DefaultCapabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetDefaultCopies sets the value of DefaultCopies for the instance
func (instance *CIM_Printer) SetPropertyDefaultCopies(value uint32) (err error) {
	return instance.SetProperty("DefaultCopies", (value))
}

// GetDefaultCopies gets the value of DefaultCopies for the instance
func (instance *CIM_Printer) GetPropertyDefaultCopies() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefaultCopies")
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

// SetDefaultLanguage sets the value of DefaultLanguage for the instance
func (instance *CIM_Printer) SetPropertyDefaultLanguage(value uint16) (err error) {
	return instance.SetProperty("DefaultLanguage", (value))
}

// GetDefaultLanguage gets the value of DefaultLanguage for the instance
func (instance *CIM_Printer) GetPropertyDefaultLanguage() (value uint16, err error) {
	retValue, err := instance.GetProperty("DefaultLanguage")
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

// SetDefaultMimeType sets the value of DefaultMimeType for the instance
func (instance *CIM_Printer) SetPropertyDefaultMimeType(value string) (err error) {
	return instance.SetProperty("DefaultMimeType", (value))
}

// GetDefaultMimeType gets the value of DefaultMimeType for the instance
func (instance *CIM_Printer) GetPropertyDefaultMimeType() (value string, err error) {
	retValue, err := instance.GetProperty("DefaultMimeType")
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

// SetDefaultNumberUp sets the value of DefaultNumberUp for the instance
func (instance *CIM_Printer) SetPropertyDefaultNumberUp(value uint32) (err error) {
	return instance.SetProperty("DefaultNumberUp", (value))
}

// GetDefaultNumberUp gets the value of DefaultNumberUp for the instance
func (instance *CIM_Printer) GetPropertyDefaultNumberUp() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefaultNumberUp")
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

// SetDefaultPaperType sets the value of DefaultPaperType for the instance
func (instance *CIM_Printer) SetPropertyDefaultPaperType(value string) (err error) {
	return instance.SetProperty("DefaultPaperType", (value))
}

// GetDefaultPaperType gets the value of DefaultPaperType for the instance
func (instance *CIM_Printer) GetPropertyDefaultPaperType() (value string, err error) {
	retValue, err := instance.GetProperty("DefaultPaperType")
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

// SetDetectedErrorState sets the value of DetectedErrorState for the instance
func (instance *CIM_Printer) SetPropertyDetectedErrorState(value uint16) (err error) {
	return instance.SetProperty("DetectedErrorState", (value))
}

// GetDetectedErrorState gets the value of DetectedErrorState for the instance
func (instance *CIM_Printer) GetPropertyDetectedErrorState() (value uint16, err error) {
	retValue, err := instance.GetProperty("DetectedErrorState")
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

// SetErrorInformation sets the value of ErrorInformation for the instance
func (instance *CIM_Printer) SetPropertyErrorInformation(value []string) (err error) {
	return instance.SetProperty("ErrorInformation", (value))
}

// GetErrorInformation gets the value of ErrorInformation for the instance
func (instance *CIM_Printer) GetPropertyErrorInformation() (value []string, err error) {
	retValue, err := instance.GetProperty("ErrorInformation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *CIM_Printer) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *CIM_Printer) GetPropertyHorizontalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("HorizontalResolution")
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

// SetJobCountSinceLastReset sets the value of JobCountSinceLastReset for the instance
func (instance *CIM_Printer) SetPropertyJobCountSinceLastReset(value uint32) (err error) {
	return instance.SetProperty("JobCountSinceLastReset", (value))
}

// GetJobCountSinceLastReset gets the value of JobCountSinceLastReset for the instance
func (instance *CIM_Printer) GetPropertyJobCountSinceLastReset() (value uint32, err error) {
	retValue, err := instance.GetProperty("JobCountSinceLastReset")
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

// SetLanguagesSupported sets the value of LanguagesSupported for the instance
func (instance *CIM_Printer) SetPropertyLanguagesSupported(value []uint16) (err error) {
	return instance.SetProperty("LanguagesSupported", (value))
}

// GetLanguagesSupported gets the value of LanguagesSupported for the instance
func (instance *CIM_Printer) GetPropertyLanguagesSupported() (value []uint16, err error) {
	retValue, err := instance.GetProperty("LanguagesSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetMarkingTechnology sets the value of MarkingTechnology for the instance
func (instance *CIM_Printer) SetPropertyMarkingTechnology(value uint16) (err error) {
	return instance.SetProperty("MarkingTechnology", (value))
}

// GetMarkingTechnology gets the value of MarkingTechnology for the instance
func (instance *CIM_Printer) GetPropertyMarkingTechnology() (value uint16, err error) {
	retValue, err := instance.GetProperty("MarkingTechnology")
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

// SetMaxCopies sets the value of MaxCopies for the instance
func (instance *CIM_Printer) SetPropertyMaxCopies(value uint32) (err error) {
	return instance.SetProperty("MaxCopies", (value))
}

// GetMaxCopies gets the value of MaxCopies for the instance
func (instance *CIM_Printer) GetPropertyMaxCopies() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxCopies")
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

// SetMaxNumberUp sets the value of MaxNumberUp for the instance
func (instance *CIM_Printer) SetPropertyMaxNumberUp(value uint32) (err error) {
	return instance.SetProperty("MaxNumberUp", (value))
}

// GetMaxNumberUp gets the value of MaxNumberUp for the instance
func (instance *CIM_Printer) GetPropertyMaxNumberUp() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxNumberUp")
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

// SetMaxSizeSupported sets the value of MaxSizeSupported for the instance
func (instance *CIM_Printer) SetPropertyMaxSizeSupported(value uint32) (err error) {
	return instance.SetProperty("MaxSizeSupported", (value))
}

// GetMaxSizeSupported gets the value of MaxSizeSupported for the instance
func (instance *CIM_Printer) GetPropertyMaxSizeSupported() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxSizeSupported")
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

// SetMimeTypesSupported sets the value of MimeTypesSupported for the instance
func (instance *CIM_Printer) SetPropertyMimeTypesSupported(value []string) (err error) {
	return instance.SetProperty("MimeTypesSupported", (value))
}

// GetMimeTypesSupported gets the value of MimeTypesSupported for the instance
func (instance *CIM_Printer) GetPropertyMimeTypesSupported() (value []string, err error) {
	retValue, err := instance.GetProperty("MimeTypesSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetNaturalLanguagesSupported sets the value of NaturalLanguagesSupported for the instance
func (instance *CIM_Printer) SetPropertyNaturalLanguagesSupported(value []string) (err error) {
	return instance.SetProperty("NaturalLanguagesSupported", (value))
}

// GetNaturalLanguagesSupported gets the value of NaturalLanguagesSupported for the instance
func (instance *CIM_Printer) GetPropertyNaturalLanguagesSupported() (value []string, err error) {
	retValue, err := instance.GetProperty("NaturalLanguagesSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetPaperSizesSupported sets the value of PaperSizesSupported for the instance
func (instance *CIM_Printer) SetPropertyPaperSizesSupported(value []uint16) (err error) {
	return instance.SetProperty("PaperSizesSupported", (value))
}

// GetPaperSizesSupported gets the value of PaperSizesSupported for the instance
func (instance *CIM_Printer) GetPropertyPaperSizesSupported() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PaperSizesSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetPaperTypesAvailable sets the value of PaperTypesAvailable for the instance
func (instance *CIM_Printer) SetPropertyPaperTypesAvailable(value []string) (err error) {
	return instance.SetProperty("PaperTypesAvailable", (value))
}

// GetPaperTypesAvailable gets the value of PaperTypesAvailable for the instance
func (instance *CIM_Printer) GetPropertyPaperTypesAvailable() (value []string, err error) {
	retValue, err := instance.GetProperty("PaperTypesAvailable")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetPrinterStatus sets the value of PrinterStatus for the instance
func (instance *CIM_Printer) SetPropertyPrinterStatus(value uint16) (err error) {
	return instance.SetProperty("PrinterStatus", (value))
}

// GetPrinterStatus gets the value of PrinterStatus for the instance
func (instance *CIM_Printer) GetPropertyPrinterStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("PrinterStatus")
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

// SetTimeOfLastReset sets the value of TimeOfLastReset for the instance
func (instance *CIM_Printer) SetPropertyTimeOfLastReset(value string) (err error) {
	return instance.SetProperty("TimeOfLastReset", (value))
}

// GetTimeOfLastReset gets the value of TimeOfLastReset for the instance
func (instance *CIM_Printer) GetPropertyTimeOfLastReset() (value string, err error) {
	retValue, err := instance.GetProperty("TimeOfLastReset")
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

// SetVerticalResolution sets the value of VerticalResolution for the instance
func (instance *CIM_Printer) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *CIM_Printer) GetPropertyVerticalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("VerticalResolution")
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
