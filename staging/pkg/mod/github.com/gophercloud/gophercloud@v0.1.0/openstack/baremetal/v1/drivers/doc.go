/*
Package drivers contains the functionality for Listing drivers, driver details,
driver properties and driver logical disk properties

API reference: https://developer.openstack.org/api-ref/baremetal/#drivers-drivers

Example to List Drivers

	drivers.ListDrivers(client.ServiceClient(), drivers.ListDriversOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		driversList, err := drivers.ExtractDrivers(page)
		if err != nil {
			return false, err
		}

		for _, n := range driversList {
			// Do something
		}

		return true, nil
	})

Example to Get single Driver Details

	showDriverDetails, err := drivers.GetDriverDetails(client, "ipmi").Extract()
	if err != nil {
		panic(err)
	}

Example to Get single Driver Properties

	showDriverProperties, err := drivers.GetDriverProperties(client, "ipmi").Extract()
	if err != nil {
		panic(err)
	}

Example to Get single Driver Logical Disk Properties

	showDriverDiskProperties, err := drivers.GetDriverDiskProperties(client, "ipmi").Extract()
	if err != nil {
		panic(err)
	}
*/
package drivers
