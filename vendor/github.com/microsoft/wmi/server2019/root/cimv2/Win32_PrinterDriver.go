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

// Win32_PrinterDriver struct
type Win32_PrinterDriver struct {
	*CIM_Service

	// The ConfigFile property contains the configuration file for this printer driver, (example: pscrptui.dll).
	ConfigFile string

	// The DataFile property contains the data file for this printer driver, (example: qms810.ppd).
	DataFile string

	// The DefaultDataType property indicates the default data type for this printer driver, (example: EMF).
	DefaultDataType string

	// The DependentFiles property contains a list of dependent files for this printer driver.
	DependentFiles []string

	// The DriverPath property contains the path for this printer driver, (example: C:\drivers\pscript.dll).
	DriverPath string

	// The FilePath property contains the path to the INF file being used, (Example: c:\temp\driver).
	FilePath string

	// The HelpFile property contains the help file for this printer driver, (example: pscrptui.hlp).
	HelpFile string

	// The InfName property contains the name of the INF file being used. The default is 'ntprint.INF'.  This will only be different if the drivers are provided directly by the manufacturer of the printer and not the operating system.
	InfName string

	// The MonitorName property contains the name of the of the monitor for this printer driver, (example: PJL monitor).
	MonitorName string

	// The OEMUrl property provides a world wide web link to the printer manufacturer's web site.  Note that this property is not populated when the Win32.INF file is used and is only applicable for drivers provided directly from the manufacturer.
	OEMUrl string

	// The SupportedPlatform property indicates the operating environments that the driver is intended for.  Examples are 'Windows NT x86' or 'Windows IA64'.
	SupportedPlatform string

	// The Version property indicates the operating system version that the driver is intended for.
	Version PrinterDriver_Version
}

func NewWin32_PrinterDriverEx1(instance *cim.WmiInstance) (newInstance *Win32_PrinterDriver, err error) {
	tmp, err := NewCIM_ServiceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PrinterDriver{
		CIM_Service: tmp,
	}
	return
}

func NewWin32_PrinterDriverEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PrinterDriver, err error) {
	tmp, err := NewCIM_ServiceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PrinterDriver{
		CIM_Service: tmp,
	}
	return
}

// SetConfigFile sets the value of ConfigFile for the instance
func (instance *Win32_PrinterDriver) SetPropertyConfigFile(value string) (err error) {
	return instance.SetProperty("ConfigFile", (value))
}

// GetConfigFile gets the value of ConfigFile for the instance
func (instance *Win32_PrinterDriver) GetPropertyConfigFile() (value string, err error) {
	retValue, err := instance.GetProperty("ConfigFile")
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

// SetDataFile sets the value of DataFile for the instance
func (instance *Win32_PrinterDriver) SetPropertyDataFile(value string) (err error) {
	return instance.SetProperty("DataFile", (value))
}

// GetDataFile gets the value of DataFile for the instance
func (instance *Win32_PrinterDriver) GetPropertyDataFile() (value string, err error) {
	retValue, err := instance.GetProperty("DataFile")
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

// SetDefaultDataType sets the value of DefaultDataType for the instance
func (instance *Win32_PrinterDriver) SetPropertyDefaultDataType(value string) (err error) {
	return instance.SetProperty("DefaultDataType", (value))
}

// GetDefaultDataType gets the value of DefaultDataType for the instance
func (instance *Win32_PrinterDriver) GetPropertyDefaultDataType() (value string, err error) {
	retValue, err := instance.GetProperty("DefaultDataType")
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

// SetDependentFiles sets the value of DependentFiles for the instance
func (instance *Win32_PrinterDriver) SetPropertyDependentFiles(value []string) (err error) {
	return instance.SetProperty("DependentFiles", (value))
}

// GetDependentFiles gets the value of DependentFiles for the instance
func (instance *Win32_PrinterDriver) GetPropertyDependentFiles() (value []string, err error) {
	retValue, err := instance.GetProperty("DependentFiles")
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

// SetDriverPath sets the value of DriverPath for the instance
func (instance *Win32_PrinterDriver) SetPropertyDriverPath(value string) (err error) {
	return instance.SetProperty("DriverPath", (value))
}

// GetDriverPath gets the value of DriverPath for the instance
func (instance *Win32_PrinterDriver) GetPropertyDriverPath() (value string, err error) {
	retValue, err := instance.GetProperty("DriverPath")
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

// SetFilePath sets the value of FilePath for the instance
func (instance *Win32_PrinterDriver) SetPropertyFilePath(value string) (err error) {
	return instance.SetProperty("FilePath", (value))
}

// GetFilePath gets the value of FilePath for the instance
func (instance *Win32_PrinterDriver) GetPropertyFilePath() (value string, err error) {
	retValue, err := instance.GetProperty("FilePath")
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

// SetHelpFile sets the value of HelpFile for the instance
func (instance *Win32_PrinterDriver) SetPropertyHelpFile(value string) (err error) {
	return instance.SetProperty("HelpFile", (value))
}

// GetHelpFile gets the value of HelpFile for the instance
func (instance *Win32_PrinterDriver) GetPropertyHelpFile() (value string, err error) {
	retValue, err := instance.GetProperty("HelpFile")
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

// SetInfName sets the value of InfName for the instance
func (instance *Win32_PrinterDriver) SetPropertyInfName(value string) (err error) {
	return instance.SetProperty("InfName", (value))
}

// GetInfName gets the value of InfName for the instance
func (instance *Win32_PrinterDriver) GetPropertyInfName() (value string, err error) {
	retValue, err := instance.GetProperty("InfName")
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

// SetMonitorName sets the value of MonitorName for the instance
func (instance *Win32_PrinterDriver) SetPropertyMonitorName(value string) (err error) {
	return instance.SetProperty("MonitorName", (value))
}

// GetMonitorName gets the value of MonitorName for the instance
func (instance *Win32_PrinterDriver) GetPropertyMonitorName() (value string, err error) {
	retValue, err := instance.GetProperty("MonitorName")
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

// SetOEMUrl sets the value of OEMUrl for the instance
func (instance *Win32_PrinterDriver) SetPropertyOEMUrl(value string) (err error) {
	return instance.SetProperty("OEMUrl", (value))
}

// GetOEMUrl gets the value of OEMUrl for the instance
func (instance *Win32_PrinterDriver) GetPropertyOEMUrl() (value string, err error) {
	retValue, err := instance.GetProperty("OEMUrl")
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

// SetSupportedPlatform sets the value of SupportedPlatform for the instance
func (instance *Win32_PrinterDriver) SetPropertySupportedPlatform(value string) (err error) {
	return instance.SetProperty("SupportedPlatform", (value))
}

// GetSupportedPlatform gets the value of SupportedPlatform for the instance
func (instance *Win32_PrinterDriver) GetPropertySupportedPlatform() (value string, err error) {
	retValue, err := instance.GetProperty("SupportedPlatform")
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

// SetVersion sets the value of Version for the instance
func (instance *Win32_PrinterDriver) SetPropertyVersion(value PrinterDriver_Version) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *Win32_PrinterDriver) GetPropertyVersion() (value PrinterDriver_Version, err error) {
	retValue, err := instance.GetProperty("Version")
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

	value = PrinterDriver_Version(valuetmp)

	return
}

// The AddPrinterDriver method installs a printer driver. The method can return the following values:
///0 - Success.
///5 - Access denied.
///1797 - The printer driver is unknown.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="DriverInfo" type="Win32_PrinterDriver ">The DriverInfo parameter specifies the neccessary inforation needed in order to create the printer driver.</param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_PrinterDriver) AddPrinterDriver( /* IN */ DriverInfo Win32_PrinterDriver) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AddPrinterDriver", DriverInfo)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
