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

// Win32_Product struct
type Win32_Product struct {
	*CIM_Product

	//
	AssignmentType uint16

	//
	HelpLink string

	//
	HelpTelephone string

	//
	InstallDate string

	//
	InstallDate2 string

	//
	InstallLocation string

	//
	InstallSource string

	//
	InstallState int16

	//
	Language string

	//
	LocalPackage string

	//
	PackageCache string

	//
	PackageCode string

	//
	PackageName string

	//
	ProductID string

	//
	RegCompany string

	//
	RegOwner string

	//
	Transforms string

	//
	URLInfoAbout string

	//
	URLUpdateInfo string

	//
	WordCount uint32
}

func NewWin32_ProductEx1(instance *cim.WmiInstance) (newInstance *Win32_Product, err error) {
	tmp, err := NewCIM_ProductEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Product{
		CIM_Product: tmp,
	}
	return
}

func NewWin32_ProductEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Product, err error) {
	tmp, err := NewCIM_ProductEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Product{
		CIM_Product: tmp,
	}
	return
}

// SetAssignmentType sets the value of AssignmentType for the instance
func (instance *Win32_Product) SetPropertyAssignmentType(value uint16) (err error) {
	return instance.SetProperty("AssignmentType", (value))
}

// GetAssignmentType gets the value of AssignmentType for the instance
func (instance *Win32_Product) GetPropertyAssignmentType() (value uint16, err error) {
	retValue, err := instance.GetProperty("AssignmentType")
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

// SetHelpLink sets the value of HelpLink for the instance
func (instance *Win32_Product) SetPropertyHelpLink(value string) (err error) {
	return instance.SetProperty("HelpLink", (value))
}

// GetHelpLink gets the value of HelpLink for the instance
func (instance *Win32_Product) GetPropertyHelpLink() (value string, err error) {
	retValue, err := instance.GetProperty("HelpLink")
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

// SetHelpTelephone sets the value of HelpTelephone for the instance
func (instance *Win32_Product) SetPropertyHelpTelephone(value string) (err error) {
	return instance.SetProperty("HelpTelephone", (value))
}

// GetHelpTelephone gets the value of HelpTelephone for the instance
func (instance *Win32_Product) GetPropertyHelpTelephone() (value string, err error) {
	retValue, err := instance.GetProperty("HelpTelephone")
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

// SetInstallDate sets the value of InstallDate for the instance
func (instance *Win32_Product) SetPropertyInstallDate(value string) (err error) {
	return instance.SetProperty("InstallDate", (value))
}

// GetInstallDate gets the value of InstallDate for the instance
func (instance *Win32_Product) GetPropertyInstallDate() (value string, err error) {
	retValue, err := instance.GetProperty("InstallDate")
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

// SetInstallDate2 sets the value of InstallDate2 for the instance
func (instance *Win32_Product) SetPropertyInstallDate2(value string) (err error) {
	return instance.SetProperty("InstallDate2", (value))
}

// GetInstallDate2 gets the value of InstallDate2 for the instance
func (instance *Win32_Product) GetPropertyInstallDate2() (value string, err error) {
	retValue, err := instance.GetProperty("InstallDate2")
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

// SetInstallLocation sets the value of InstallLocation for the instance
func (instance *Win32_Product) SetPropertyInstallLocation(value string) (err error) {
	return instance.SetProperty("InstallLocation", (value))
}

// GetInstallLocation gets the value of InstallLocation for the instance
func (instance *Win32_Product) GetPropertyInstallLocation() (value string, err error) {
	retValue, err := instance.GetProperty("InstallLocation")
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

// SetInstallSource sets the value of InstallSource for the instance
func (instance *Win32_Product) SetPropertyInstallSource(value string) (err error) {
	return instance.SetProperty("InstallSource", (value))
}

// GetInstallSource gets the value of InstallSource for the instance
func (instance *Win32_Product) GetPropertyInstallSource() (value string, err error) {
	retValue, err := instance.GetProperty("InstallSource")
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

// SetInstallState sets the value of InstallState for the instance
func (instance *Win32_Product) SetPropertyInstallState(value int16) (err error) {
	return instance.SetProperty("InstallState", (value))
}

// GetInstallState gets the value of InstallState for the instance
func (instance *Win32_Product) GetPropertyInstallState() (value int16, err error) {
	retValue, err := instance.GetProperty("InstallState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetLanguage sets the value of Language for the instance
func (instance *Win32_Product) SetPropertyLanguage(value string) (err error) {
	return instance.SetProperty("Language", (value))
}

// GetLanguage gets the value of Language for the instance
func (instance *Win32_Product) GetPropertyLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("Language")
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

// SetLocalPackage sets the value of LocalPackage for the instance
func (instance *Win32_Product) SetPropertyLocalPackage(value string) (err error) {
	return instance.SetProperty("LocalPackage", (value))
}

// GetLocalPackage gets the value of LocalPackage for the instance
func (instance *Win32_Product) GetPropertyLocalPackage() (value string, err error) {
	retValue, err := instance.GetProperty("LocalPackage")
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

// SetPackageCache sets the value of PackageCache for the instance
func (instance *Win32_Product) SetPropertyPackageCache(value string) (err error) {
	return instance.SetProperty("PackageCache", (value))
}

// GetPackageCache gets the value of PackageCache for the instance
func (instance *Win32_Product) GetPropertyPackageCache() (value string, err error) {
	retValue, err := instance.GetProperty("PackageCache")
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

// SetPackageCode sets the value of PackageCode for the instance
func (instance *Win32_Product) SetPropertyPackageCode(value string) (err error) {
	return instance.SetProperty("PackageCode", (value))
}

// GetPackageCode gets the value of PackageCode for the instance
func (instance *Win32_Product) GetPropertyPackageCode() (value string, err error) {
	retValue, err := instance.GetProperty("PackageCode")
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

// SetPackageName sets the value of PackageName for the instance
func (instance *Win32_Product) SetPropertyPackageName(value string) (err error) {
	return instance.SetProperty("PackageName", (value))
}

// GetPackageName gets the value of PackageName for the instance
func (instance *Win32_Product) GetPropertyPackageName() (value string, err error) {
	retValue, err := instance.GetProperty("PackageName")
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

// SetProductID sets the value of ProductID for the instance
func (instance *Win32_Product) SetPropertyProductID(value string) (err error) {
	return instance.SetProperty("ProductID", (value))
}

// GetProductID gets the value of ProductID for the instance
func (instance *Win32_Product) GetPropertyProductID() (value string, err error) {
	retValue, err := instance.GetProperty("ProductID")
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

// SetRegCompany sets the value of RegCompany for the instance
func (instance *Win32_Product) SetPropertyRegCompany(value string) (err error) {
	return instance.SetProperty("RegCompany", (value))
}

// GetRegCompany gets the value of RegCompany for the instance
func (instance *Win32_Product) GetPropertyRegCompany() (value string, err error) {
	retValue, err := instance.GetProperty("RegCompany")
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

// SetRegOwner sets the value of RegOwner for the instance
func (instance *Win32_Product) SetPropertyRegOwner(value string) (err error) {
	return instance.SetProperty("RegOwner", (value))
}

// GetRegOwner gets the value of RegOwner for the instance
func (instance *Win32_Product) GetPropertyRegOwner() (value string, err error) {
	retValue, err := instance.GetProperty("RegOwner")
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

// SetTransforms sets the value of Transforms for the instance
func (instance *Win32_Product) SetPropertyTransforms(value string) (err error) {
	return instance.SetProperty("Transforms", (value))
}

// GetTransforms gets the value of Transforms for the instance
func (instance *Win32_Product) GetPropertyTransforms() (value string, err error) {
	retValue, err := instance.GetProperty("Transforms")
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

// SetURLInfoAbout sets the value of URLInfoAbout for the instance
func (instance *Win32_Product) SetPropertyURLInfoAbout(value string) (err error) {
	return instance.SetProperty("URLInfoAbout", (value))
}

// GetURLInfoAbout gets the value of URLInfoAbout for the instance
func (instance *Win32_Product) GetPropertyURLInfoAbout() (value string, err error) {
	retValue, err := instance.GetProperty("URLInfoAbout")
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

// SetURLUpdateInfo sets the value of URLUpdateInfo for the instance
func (instance *Win32_Product) SetPropertyURLUpdateInfo(value string) (err error) {
	return instance.SetProperty("URLUpdateInfo", (value))
}

// GetURLUpdateInfo gets the value of URLUpdateInfo for the instance
func (instance *Win32_Product) GetPropertyURLUpdateInfo() (value string, err error) {
	retValue, err := instance.GetProperty("URLUpdateInfo")
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

// SetWordCount sets the value of WordCount for the instance
func (instance *Win32_Product) SetPropertyWordCount(value uint32) (err error) {
	return instance.SetProperty("WordCount", (value))
}

// GetWordCount gets the value of WordCount for the instance
func (instance *Win32_Product) GetPropertyWordCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("WordCount")
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

//

// <param name="AllUsers" type="bool "></param>
// <param name="Options" type="string "></param>
// <param name="PackageLocation" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Install( /* IN */ PackageLocation string,
	/* IN */ Options string,
	/* IN */ AllUsers bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Install", PackageLocation, Options, AllUsers)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Options" type="string "></param>
// <param name="PackageLocation" type="string "></param>
// <param name="TargetLocation" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Admin( /* IN */ PackageLocation string,
	/* IN */ TargetLocation string,
	/* IN */ Options string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Admin", PackageLocation, TargetLocation, Options)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AllUsers" type="bool "></param>
// <param name="Options" type="string "></param>
// <param name="PackageLocation" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Advertise( /* IN */ PackageLocation string,
	/* IN */ Options string,
	/* IN */ AllUsers bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Advertise", PackageLocation, Options, AllUsers)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReinstallMode" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Reinstall( /* IN */ ReinstallMode uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Reinstall", ReinstallMode)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Options" type="string "></param>
// <param name="PackageLocation" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Upgrade( /* IN */ PackageLocation string,
	/* IN */ Options string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Upgrade", PackageLocation, Options)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="InstallLevel" type="uint16 "></param>
// <param name="InstallState" type="uint16 "></param>
// <param name="Options" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Configure( /* IN */ InstallState uint16,
	/* IN */ InstallLevel uint16,
	/* IN */ Options string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Configure", InstallState, InstallLevel, Options)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Product) Uninstall() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Uninstall")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
