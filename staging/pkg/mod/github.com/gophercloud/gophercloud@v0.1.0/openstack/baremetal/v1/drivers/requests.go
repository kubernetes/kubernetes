package drivers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListDriversOptsBuilder allows extensions to add additional parameters to the
// ListDrivers request.
type ListDriversOptsBuilder interface {
	ToListDriversOptsQuery() (string, error)
}

// ListDriversOpts defines query options that can be passed to ListDrivers
type ListDriversOpts struct {
	// Provide detailed information about the drivers
	Detail bool `q:"detail"`

	// Filter the list by the type of the driver
	Type string `q:"type"`
}

// ToListDriversOptsQuery formats a ListOpts into a query string
func (opts ListDriversOpts) ToListDriversOptsQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDrivers makes a request against the API to list all drivers
func ListDrivers(client *gophercloud.ServiceClient, opts ListDriversOptsBuilder) pagination.Pager {
	url := driversURL(client)
	if opts != nil {
		query, err := opts.ToListDriversOptsQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return DriverPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// GetDriverDetails Shows details for a driver
func GetDriverDetails(client *gophercloud.ServiceClient, driverName string) (r GetDriverResult) {
	_, r.Err = client.Get(driverDetailsURL(client, driverName), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// GetDriverProperties Shows the required and optional parameters that
// driverName expects to be supplied in the driver_info field for every
// Node it manages
func GetDriverProperties(client *gophercloud.ServiceClient, driverName string) (r GetPropertiesResult) {
	_, r.Err = client.Get(driverPropertiesURL(client, driverName), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// GetDriverDiskProperties Show the required and optional parameters that
// driverName expects to be supplied in the node’s raid_config field, if a
// RAID configuration change is requested.
func GetDriverDiskProperties(client *gophercloud.ServiceClient, driverName string) (r GetDiskPropertiesResult) {
	_, r.Err = client.Get(driverDiskPropertiesURL(client, driverName), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
