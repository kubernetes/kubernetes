// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_FolderRedirectionUserConfiguration struct
type Win32_FolderRedirectionUserConfiguration struct {
	*cim.WmiInstance

	// AppData\Roaming folder, GUID is {3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}
	AppDataRoaming Win32_FolderRedirection

	// Contacts folder, GUID is {56784854-C6CB-462b-8169-88E350ACB882}
	Contacts Win32_FolderRedirection

	// Desktop folder, GUID is {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
	Desktop Win32_FolderRedirection

	// Documents folder, GUID is {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
	Documents Win32_FolderRedirection

	// Downloads folder, GUID is {374DE290-123F-4565-9164-39C4925E467B}
	Downloads Win32_FolderRedirection

	// Favorites folder, GUID is {1777F761-68AD-4D8A-87BD-30B759FA33DD}
	Favorites Win32_FolderRedirection

	// Indicates if the settings configured through this WMI class are taking affect.
	IsConfiguredByWMI bool

	// Links folder, GUID is {BFB9D5E0-C6A9-404C-B2B2-AE6DB6AF4968}
	Links Win32_FolderRedirection

	// Music folder, GUID is {4BD8D571-6D19-48D3-BE97-422220080E43}
	Music Win32_FolderRedirection

	// Pictures folder, GUID is {33E28130-4E1E-4676-835A-98395C3BC3BB}
	Pictures Win32_FolderRedirection

	// The Primary Computer feature is enabled for this user
	PrimaryComputerEnabled bool

	// SavedGames folder, GUID is {4C5C32FF-BB9D-43b0-B5B4-2D72E54EAAA4}
	SavedGames Win32_FolderRedirection

	// Searches folder, GUID is {7D1D3A04-DEBB-4115-95CF-2F29DA2920DA}
	Searches Win32_FolderRedirection

	// Start Menu folder, GUID is {625B53C3-AB48-4EC1-BA1F-A1EF4146FC19}
	StartMenu Win32_FolderRedirection

	// Videos folder, GUID is {18989B1D-99B5-455B-841C-AB7C74E4DDFC}
	Videos Win32_FolderRedirection
}

func NewWin32_FolderRedirectionUserConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_FolderRedirectionUserConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_FolderRedirectionUserConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_FolderRedirectionUserConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetAppDataRoaming sets the value of AppDataRoaming for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyAppDataRoaming(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("AppDataRoaming", (value))
}

// GetAppDataRoaming gets the value of AppDataRoaming for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyAppDataRoaming() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("AppDataRoaming")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetContacts sets the value of Contacts for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyContacts(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Contacts", (value))
}

// GetContacts gets the value of Contacts for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyContacts() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Contacts")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetDesktop sets the value of Desktop for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyDesktop(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Desktop", (value))
}

// GetDesktop gets the value of Desktop for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyDesktop() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Desktop")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetDocuments sets the value of Documents for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyDocuments(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Documents", (value))
}

// GetDocuments gets the value of Documents for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyDocuments() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Documents")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetDownloads sets the value of Downloads for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyDownloads(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Downloads", (value))
}

// GetDownloads gets the value of Downloads for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyDownloads() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Downloads")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetFavorites sets the value of Favorites for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyFavorites(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Favorites", (value))
}

// GetFavorites gets the value of Favorites for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyFavorites() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Favorites")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetIsConfiguredByWMI sets the value of IsConfiguredByWMI for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyIsConfiguredByWMI(value bool) (err error) {
	return instance.SetProperty("IsConfiguredByWMI", (value))
}

// GetIsConfiguredByWMI gets the value of IsConfiguredByWMI for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyIsConfiguredByWMI() (value bool, err error) {
	retValue, err := instance.GetProperty("IsConfiguredByWMI")
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

// SetLinks sets the value of Links for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyLinks(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Links", (value))
}

// GetLinks gets the value of Links for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyLinks() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Links")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetMusic sets the value of Music for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyMusic(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Music", (value))
}

// GetMusic gets the value of Music for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyMusic() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Music")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetPictures sets the value of Pictures for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyPictures(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Pictures", (value))
}

// GetPictures gets the value of Pictures for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyPictures() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Pictures")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetPrimaryComputerEnabled sets the value of PrimaryComputerEnabled for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyPrimaryComputerEnabled(value bool) (err error) {
	return instance.SetProperty("PrimaryComputerEnabled", (value))
}

// GetPrimaryComputerEnabled gets the value of PrimaryComputerEnabled for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyPrimaryComputerEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("PrimaryComputerEnabled")
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

// SetSavedGames sets the value of SavedGames for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertySavedGames(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("SavedGames", (value))
}

// GetSavedGames gets the value of SavedGames for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertySavedGames() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("SavedGames")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetSearches sets the value of Searches for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertySearches(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Searches", (value))
}

// GetSearches gets the value of Searches for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertySearches() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Searches")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetStartMenu sets the value of StartMenu for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyStartMenu(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("StartMenu", (value))
}

// GetStartMenu gets the value of StartMenu for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyStartMenu() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("StartMenu")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}

// SetVideos sets the value of Videos for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) SetPropertyVideos(value Win32_FolderRedirection) (err error) {
	return instance.SetProperty("Videos", (value))
}

// GetVideos gets the value of Videos for the instance
func (instance *Win32_FolderRedirectionUserConfiguration) GetPropertyVideos() (value Win32_FolderRedirection, err error) {
	retValue, err := instance.GetProperty("Videos")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_FolderRedirection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_FolderRedirection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_FolderRedirection(valuetmp)

	return
}
