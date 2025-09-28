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

// Win32_PrinterConfiguration struct
type Win32_PrinterConfiguration struct {
	*CIM_Setting

	// The BitsPerPel property contains the number of bits per pixel for the output device Win32 printer.  This member is used by display drivers and not by printer drivers.
	///Example: 8.
	/// This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	BitsPerPel uint32

	// The Collate property specifies whether to collate the pages that are printed. To collate is to print out the entire document before printing the next copy, as opposed to printing out each page of the document the required number times. This property is ignored unless the printer driver indicates support for collation.
	///Values: TRUE or FALSE. If TRUE, the printer collates all documents.
	Collate bool

	// The Color property indicates whether the document is to be printed in color or monochrome.  Some color printers have the capability to print using true black instead of a combination of Yellow, Cyan, and Magenta.  This usually creates darker and sharper text for documents.  This option is only useful for color printers that support true black printing.
	Color PrinterConfiguration_Color

	// The Copies property indicates the number of copies to be printed. The printer driver must support printing multi-page copies.
	///Example: 2
	Copies uint32

	// The DeviceName property specifies the friendly name of the printer.  This name is unique to the type of printer and may be truncated because of the limitations of the string from which it is derived.
	///Example PCL/HP LaserJet
	DeviceName string

	// The DisplayFlags property contains two bits of information about the display. This member communicates whether the display device is monochrome or colored, and interlaced or non-interlaced, by masking its value with the DM_GRAYSCALE and DM_INTERLACED masks respectively.
	///This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	DisplayFlags uint32

	// The DisplayFrequency property indicates the refresh frequency of the display The refresh frequency for a monitor is the number of times the screen is redrawn per second.
	///This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	DisplayFrequency uint32

	// The DitherType property indicates the dither type of the printer.  This member can assume predefined values of 1 to 5, or driver-defined values from 6 to 256.  Line art dithering is a special dithering method that produces well defined borders between black, white, and gray scalings.  It is not suitable for images that include continuous graduations in intensity and hue such as scanned photographs.
	DitherType PrinterConfiguration_DitherType

	// The DriverVersion property indicates the version number of the Win32 printer driver.  The version numbers are created and maintained by the driver manufacturer.
	DriverVersion uint32

	// The Duplex property indicates whether printing is done on one or both sides.
	///Values: TRUE or FALSE. If TRUE, printing is done on both sides.
	Duplex bool

	// The FormName property indicates the name of the form used for the print job.  This property is used only on Windows NT/Windows 2000 systems.
	///Example: Legal
	FormName string

	// The HorizontalResolution property indicates the print resolution along the X axis (width) of the print job. This value is only set when the PrintQuality property of this class is positive and is similar to the XResolution property.
	HorizontalResolution uint32

	// The ICMIntent (Image Color Matching Intent) property indicates the specific value of one of the three possible color matching methods (called intents) that should be used by default.  ICM applications establish intents by using the ICM functions.  This property can assume predefined values of 1 to 3, or driver-defined values from 4 to 256.  Non-ICM applications can use this value to determine how the printer handles color printing jobs.
	ICMIntent PrinterConfiguration_ICMIntent

	// The ICMMethod (Image Color Matching Method) property specifies how ICM is handled.  For a non-ICM application, this property determines if ICM is enabled or disabled.  For ICM applications, the system examines this property to determine which part of the computer system handles ICM support.
	ICMMethod PrinterConfiguration_ICMMethod

	// The LogPixels property contains the number of pixels per logical inch.  This member is valid only with devices that work with pixels (this excludes devices such as printers).
	///This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	LogPixels uint32

	// The MediaType property specifies the type of media being printed on. The property can be set to a predefined value or a driver-defined value greater than or equal to 256. For Windows 95 and later; Windows 2000.
	MediaType PrinterConfiguration_MediaType

	// The Name property indicates the name of the printer with which this configuration is associated.
	Name string

	// The Orientation property indicates the printing orientation of the paper.
	Orientation PrinterConfiguration_Orientation

	// The PaperLength property indicates the length of the paper.
	///Example: 2794
	PaperLength uint32

	// The PaperSize property indicates the size of the paper.
	///Example: A4 or Letter
	PaperSize string

	// The PaperWidth property indicates the width of the paper.
	///Example: 2159
	PaperWidth uint32

	// The PelsHeight property indicates the height of the displayable surface.
	///This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	PelsHeight uint32

	// The PelsWidth property indicates the width of the displayable surface.
	///This property has been deprecated because it is not applicable to printers.  There is no replacement value.
	PelsWidth uint32

	// The PrintQuality property indicates one of four quality levels of the print job.  If a positive value is specified, the quality is measured in dots per inch.
	///Example: Draft
	PrintQuality PrinterConfiguration_PrintQuality

	// The Scale property specifies the factor by which the printed output is to be scaled.  For example a scale of 75 reduces the print output to 3/4 its original height and width.
	Scale uint32

	// The SpecificationVersion property indicates the version number of the initialization data for the device associated with the Win32 printer.
	SpecificationVersion uint32

	// The TTOption property specifies how TrueType(r) fonts should be printed.  There are 3 possible values:
	///Bitmap -  Prints TrueType fonts as graphics. This is the default action for dot-matrix printers.
	///Download -  Downloads TrueType fonts as soft fonts. This is the default action for printers that use the Printer Control Language (PCL).
	///Substitute -  Substitutes device fonts for TrueType fonts. This is the default action for PostScript(r) printers.
	TTOption PrinterConfiguration_TTOption

	// The VerticalResolution property indicates the print resolution along the Y axis (height) of the print job. This value is only set when the PrintQuality property of this class is positive, and is similar to the YResolution property.
	VerticalResolution uint32

	// The XResolution property has been deprecated to theHorizontalResolution property.  Please refer to the description of that property.
	XResolution uint32

	// The YResolution property has been deprecated to theVerticalResolution property.  Please refer to the description of that property.
	YResolution uint32
}

func NewWin32_PrinterConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_PrinterConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PrinterConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_PrinterConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PrinterConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PrinterConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetBitsPerPel sets the value of BitsPerPel for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyBitsPerPel(value uint32) (err error) {
	return instance.SetProperty("BitsPerPel", (value))
}

// GetBitsPerPel gets the value of BitsPerPel for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyBitsPerPel() (value uint32, err error) {
	retValue, err := instance.GetProperty("BitsPerPel")
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

// SetCollate sets the value of Collate for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyCollate(value bool) (err error) {
	return instance.SetProperty("Collate", (value))
}

// GetCollate gets the value of Collate for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyCollate() (value bool, err error) {
	retValue, err := instance.GetProperty("Collate")
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

// SetColor sets the value of Color for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyColor(value PrinterConfiguration_Color) (err error) {
	return instance.SetProperty("Color", (value))
}

// GetColor gets the value of Color for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyColor() (value PrinterConfiguration_Color, err error) {
	retValue, err := instance.GetProperty("Color")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_Color(valuetmp)

	return
}

// SetCopies sets the value of Copies for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyCopies(value uint32) (err error) {
	return instance.SetProperty("Copies", (value))
}

// GetCopies gets the value of Copies for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyCopies() (value uint32, err error) {
	retValue, err := instance.GetProperty("Copies")
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

// SetDeviceName sets the value of DeviceName for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDeviceName(value string) (err error) {
	return instance.SetProperty("DeviceName", (value))
}

// GetDeviceName gets the value of DeviceName for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDeviceName() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceName")
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

// SetDisplayFlags sets the value of DisplayFlags for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDisplayFlags(value uint32) (err error) {
	return instance.SetProperty("DisplayFlags", (value))
}

// GetDisplayFlags gets the value of DisplayFlags for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDisplayFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("DisplayFlags")
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

// SetDisplayFrequency sets the value of DisplayFrequency for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDisplayFrequency(value uint32) (err error) {
	return instance.SetProperty("DisplayFrequency", (value))
}

// GetDisplayFrequency gets the value of DisplayFrequency for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDisplayFrequency() (value uint32, err error) {
	retValue, err := instance.GetProperty("DisplayFrequency")
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

// SetDitherType sets the value of DitherType for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDitherType(value PrinterConfiguration_DitherType) (err error) {
	return instance.SetProperty("DitherType", (value))
}

// GetDitherType gets the value of DitherType for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDitherType() (value PrinterConfiguration_DitherType, err error) {
	retValue, err := instance.GetProperty("DitherType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_DitherType(valuetmp)

	return
}

// SetDriverVersion sets the value of DriverVersion for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDriverVersion(value uint32) (err error) {
	return instance.SetProperty("DriverVersion", (value))
}

// GetDriverVersion gets the value of DriverVersion for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDriverVersion() (value uint32, err error) {
	retValue, err := instance.GetProperty("DriverVersion")
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

// SetDuplex sets the value of Duplex for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyDuplex(value bool) (err error) {
	return instance.SetProperty("Duplex", (value))
}

// GetDuplex gets the value of Duplex for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyDuplex() (value bool, err error) {
	retValue, err := instance.GetProperty("Duplex")
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

// SetFormName sets the value of FormName for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyFormName(value string) (err error) {
	return instance.SetProperty("FormName", (value))
}

// GetFormName gets the value of FormName for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyFormName() (value string, err error) {
	retValue, err := instance.GetProperty("FormName")
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

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyHorizontalResolution() (value uint32, err error) {
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

// SetICMIntent sets the value of ICMIntent for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyICMIntent(value PrinterConfiguration_ICMIntent) (err error) {
	return instance.SetProperty("ICMIntent", (value))
}

// GetICMIntent gets the value of ICMIntent for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyICMIntent() (value PrinterConfiguration_ICMIntent, err error) {
	retValue, err := instance.GetProperty("ICMIntent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_ICMIntent(valuetmp)

	return
}

// SetICMMethod sets the value of ICMMethod for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyICMMethod(value PrinterConfiguration_ICMMethod) (err error) {
	return instance.SetProperty("ICMMethod", (value))
}

// GetICMMethod gets the value of ICMMethod for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyICMMethod() (value PrinterConfiguration_ICMMethod, err error) {
	retValue, err := instance.GetProperty("ICMMethod")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_ICMMethod(valuetmp)

	return
}

// SetLogPixels sets the value of LogPixels for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyLogPixels(value uint32) (err error) {
	return instance.SetProperty("LogPixels", (value))
}

// GetLogPixels gets the value of LogPixels for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyLogPixels() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogPixels")
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

// SetMediaType sets the value of MediaType for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyMediaType(value PrinterConfiguration_MediaType) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyMediaType() (value PrinterConfiguration_MediaType, err error) {
	retValue, err := instance.GetProperty("MediaType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_MediaType(valuetmp)

	return
}

// SetName sets the value of Name for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetOrientation sets the value of Orientation for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyOrientation(value PrinterConfiguration_Orientation) (err error) {
	return instance.SetProperty("Orientation", (value))
}

// GetOrientation gets the value of Orientation for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyOrientation() (value PrinterConfiguration_Orientation, err error) {
	retValue, err := instance.GetProperty("Orientation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_Orientation(valuetmp)

	return
}

// SetPaperLength sets the value of PaperLength for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPaperLength(value uint32) (err error) {
	return instance.SetProperty("PaperLength", (value))
}

// GetPaperLength gets the value of PaperLength for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPaperLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("PaperLength")
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

// SetPaperSize sets the value of PaperSize for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPaperSize(value string) (err error) {
	return instance.SetProperty("PaperSize", (value))
}

// GetPaperSize gets the value of PaperSize for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPaperSize() (value string, err error) {
	retValue, err := instance.GetProperty("PaperSize")
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

// SetPaperWidth sets the value of PaperWidth for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPaperWidth(value uint32) (err error) {
	return instance.SetProperty("PaperWidth", (value))
}

// GetPaperWidth gets the value of PaperWidth for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPaperWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("PaperWidth")
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

// SetPelsHeight sets the value of PelsHeight for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPelsHeight(value uint32) (err error) {
	return instance.SetProperty("PelsHeight", (value))
}

// GetPelsHeight gets the value of PelsHeight for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPelsHeight() (value uint32, err error) {
	retValue, err := instance.GetProperty("PelsHeight")
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

// SetPelsWidth sets the value of PelsWidth for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPelsWidth(value uint32) (err error) {
	return instance.SetProperty("PelsWidth", (value))
}

// GetPelsWidth gets the value of PelsWidth for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPelsWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("PelsWidth")
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

// SetPrintQuality sets the value of PrintQuality for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyPrintQuality(value PrinterConfiguration_PrintQuality) (err error) {
	return instance.SetProperty("PrintQuality", (value))
}

// GetPrintQuality gets the value of PrintQuality for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyPrintQuality() (value PrinterConfiguration_PrintQuality, err error) {
	retValue, err := instance.GetProperty("PrintQuality")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_PrintQuality(valuetmp)

	return
}

// SetScale sets the value of Scale for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyScale(value uint32) (err error) {
	return instance.SetProperty("Scale", (value))
}

// GetScale gets the value of Scale for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyScale() (value uint32, err error) {
	retValue, err := instance.GetProperty("Scale")
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

// SetSpecificationVersion sets the value of SpecificationVersion for the instance
func (instance *Win32_PrinterConfiguration) SetPropertySpecificationVersion(value uint32) (err error) {
	return instance.SetProperty("SpecificationVersion", (value))
}

// GetSpecificationVersion gets the value of SpecificationVersion for the instance
func (instance *Win32_PrinterConfiguration) GetPropertySpecificationVersion() (value uint32, err error) {
	retValue, err := instance.GetProperty("SpecificationVersion")
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

// SetTTOption sets the value of TTOption for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyTTOption(value PrinterConfiguration_TTOption) (err error) {
	return instance.SetProperty("TTOption", (value))
}

// GetTTOption gets the value of TTOption for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyTTOption() (value PrinterConfiguration_TTOption, err error) {
	retValue, err := instance.GetProperty("TTOption")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = PrinterConfiguration_TTOption(valuetmp)

	return
}

// SetVerticalResolution sets the value of VerticalResolution for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyVerticalResolution() (value uint32, err error) {
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

// SetXResolution sets the value of XResolution for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyXResolution(value uint32) (err error) {
	return instance.SetProperty("XResolution", (value))
}

// GetXResolution gets the value of XResolution for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyXResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("XResolution")
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

// SetYResolution sets the value of YResolution for the instance
func (instance *Win32_PrinterConfiguration) SetPropertyYResolution(value uint32) (err error) {
	return instance.SetProperty("YResolution", (value))
}

// GetYResolution gets the value of YResolution for the instance
func (instance *Win32_PrinterConfiguration) GetPropertyYResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("YResolution")
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
