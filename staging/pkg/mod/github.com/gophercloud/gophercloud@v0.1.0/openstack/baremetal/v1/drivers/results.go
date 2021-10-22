package drivers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type driverResult struct {
	gophercloud.Result
}

// Extract interprets any driverResult as a Driver, if possible.
func (r driverResult) Extract() (*Driver, error) {
	var s Driver
	err := r.ExtractInto(&s)
	return &s, err
}

func (r driverResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "")
}

func ExtractDriversInto(r pagination.Page, v interface{}) error {
	return r.(DriverPage).Result.ExtractIntoSlicePtr(v, "drivers")
}

// Driver represents a driver in the OpenStack Bare Metal API.
type Driver struct {
	// Name and Identifier of the driver
	Name string `json:"name"`

	// A list of active hosts that support this driver
	Hosts []string `json:"hosts"`

	// Type of this driver (“classic” or “dynamic”)
	Type string `json:"type"`

	// The default bios interface used for a node with a dynamic driver,
	// if no bios interface is specified for the node.
	DefaultBiosInterface string `json:"default_bios_interface"`

	// The default boot interface used for a node with a dynamic driver,
	// if no boot interface is specified for the node.
	DefaultBootInterface string `json:"default_boot_interface"`

	// The default console interface used for a node with a dynamic driver,
	// if no console interface is specified for the node.
	DefaultConsoleInterface string `json:"default_console_interface"`

	// The default deploy interface used for a node with a dynamic driver,
	// if no deploy interface is specified for the node.
	DefaultDeployInterface string `json:"default_deploy_interface"`

	// The default inspection interface used for a node with a dynamic driver,
	// if no inspection interface is specified for the node.
	DefaultInspectInterface string `json:"default_inspect_interface"`

	// The default management interface used for a node with a dynamic driver,
	// if no management interface is specified for the node.
	DefaultManagementInterface string `json:"default_management_interface"`

	// The default network interface used for a node with a dynamic driver,
	// if no network interface is specified for the node.
	DefaultNetworkInterface string `json:"default_network_interface"`

	// The default power interface used for a node with a dynamic driver,
	// if no power interface is specified for the node.
	DefaultPowerInterface string `json:"default_power_interface"`

	// The default RAID interface used for a node with a dynamic driver,
	// if no RAID interface is specified for the node.
	DefaultRaidInterface string `json:"default_raid_interface"`

	// The default rescue interface used for a node with a dynamic driver,
	// if no rescue interface is specified for the node.
	DefaultRescueInterface string `json:"default_rescue_interface"`

	// The default storage interface used for a node with a dynamic driver,
	// if no storage interface is specified for the node.
	DefaultStorageInterface string `json:"default_storage_interface"`

	// The default vendor interface used for a node with a dynamic driver,
	// if no vendor interface is specified for the node.
	DefaultVendorInterface string `json:"default_vendor_interface"`

	// The enabled bios interfaces for this driver.
	EnabledBiosInterfaces []string `json:"enabled_bios_interfaces"`

	// The enabled boot interfaces for this driver.
	EnabledBootInterfaces []string `json:"enabled_boot_interfaces"`

	// The enabled console interfaces for this driver.
	EnabledConsoleInterface []string `json:"enabled_console_interfaces"`

	// The enabled deploy interfaces for this driver.
	EnabledDeployInterfaces []string `json:"enabled_deploy_interfaces"`

	// The enabled inspection interfaces for this driver.
	EnabledInspectInterfaces []string `json:"enabled_inspect_interfaces"`

	// The enabled management interfaces for this driver.
	EnabledManagementInterfaces []string `json:"enabled_management_interfaces"`

	// The enabled network interfaces for this driver.
	EnabledNetworkInterfaces []string `json:"enabled_network_interfaces"`

	// The enabled power interfaces for this driver.
	EnabledPowerInterfaces []string `json:"enabled_power_interfaces"`

	// The enabled rescue interfaces for this driver.
	EnabledRescueInterfaces []string `json:"enabled_rescue_interfaces"`

	// The enabled RAID interfaces for this driver.
	EnabledRaidInterfaces []string `json:"enabled_raid_interfaces"`

	// The enabled storage interfaces for this driver.
	EnabledStorageInterfaces []string `json:"enabled_storage_interfaces"`

	// The enabled vendor interfaces for this driver.
	EnabledVendorInterfaces []string `json:"enabled_vendor_interfaces"`

	//A list of relative links. Includes the self and bookmark links.
	Links []interface{} `json:"links"`

	// A list of links to driver properties.
	Properties []interface{} `json:"properties"`
}

// DriverPage abstracts the raw results of making a ListDrivers() request
// against the API.
type DriverPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if a page contains no Driver results.
func (r DriverPage) IsEmpty() (bool, error) {
	s, err := ExtractDrivers(r)
	return len(s) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (r DriverPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"drivers_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractDrivers interprets the results of a single page from ListDrivers()
// call, producing a slice of Driver entities.
func ExtractDrivers(r pagination.Page) ([]Driver, error) {
	var s []Driver
	err := ExtractDriversInto(r, &s)
	return s, err
}

// GetDriverResult is the response from a Get operation.
// Call its Extract method to interpret it as a Driver.
type GetDriverResult struct {
	driverResult
}

// DriverProperties represents driver properties in the OpenStack Bare Metal API.
type DriverProperties map[string]interface{}

// Extract interprets any GetPropertiesResult as DriverProperties, if possible.
func (r GetPropertiesResult) Extract() (*DriverProperties, error) {
	var s DriverProperties
	err := r.ExtractInto(&s)
	return &s, err
}

// GetPropertiesResult is the response from a GetDriverProperties operation.
// Call its Extract method to interpret it as DriverProperties.
type GetPropertiesResult struct {
	gophercloud.Result
}

// DiskProperties represents driver disk properties in the OpenStack Bare Metal API.
type DiskProperties map[string]interface{}

// Extract interprets any GetDiskPropertiesResult as DiskProperties, if possible.
func (r GetDiskPropertiesResult) Extract() (*DiskProperties, error) {
	var s DiskProperties
	err := r.ExtractInto(&s)
	return &s, err
}

// GetDiskPropertiesResult is the response from a GetDriverDiskProperties operation.
// Call its Extract method to interpret it as DiskProperties.
type GetDiskPropertiesResult struct {
	gophercloud.Result
}
