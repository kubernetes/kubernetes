/*
Package tenantnetworks provides the ability for tenants to see information
about the networks they have access to.

This is a deprecated API and will be removed from the Nova API service in a
future version.

This API works in both Neutron and nova-network based OpenStack clouds.

Example to List Networks Available to a Tenant

	allPages, err := tenantnetworks.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allNetworks, err := tenantnetworks.ExtractNetworks(allPages)
	if err != nil {
		panic(err)
	}

	for _, network := range allNetworks {
		fmt.Printf("%+v\n", network)
	}
*/
package tenantnetworks
