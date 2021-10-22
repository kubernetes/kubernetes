/*
Package zones provides information and interaction with the zone API
resource for the OpenStack DNS service.

Example to List Zones

	listOpts := zones.ListOpts{
		Email: "jdoe@example.com",
	}

	allPages, err := zones.List(dnsClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allZones, err := zones.ExtractZones(allPages)
	if err != nil {
		panic(err)
	}

	for _, zone := range allZones {
		fmt.Printf("%+v\n", zone)
	}

Example to Create a Zone

	createOpts := zones.CreateOpts{
		Name:        "example.com.",
		Email:       "jdoe@example.com",
		Type:        "PRIMARY",
		TTL:         7200,
		Description: "This is a zone.",
	}

	zone, err := zones.Create(dnsClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Zone

	zoneID := "99d10f68-5623-4491-91a0-6daafa32b60e"
	err := zones.Delete(dnsClient, zoneID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package zones
