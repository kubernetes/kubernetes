/*
Package endpoints provides information and interaction with the service
endpoints API resource in the OpenStack Identity service.

For more information, see:
http://developer.openstack.org/api-ref-identity-v3.html#endpoints-v3

Example to List Endpoints

	serviceID := "e629d6e599d9489fb3ae5d9cc12eaea3"

	listOpts := endpoints.ListOpts{
		ServiceID: serviceID,
	}

	allPages, err := endpoints.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allEndpoints, err := endpoints.ExtractEndpoints(allPages)
	if err != nil {
		panic(err)
	}

	for _, endpoint := range allEndpoints {
		fmt.Printf("%+v\n", endpoint)
	}

Example to Create an Endpoint

	serviceID := "e629d6e599d9489fb3ae5d9cc12eaea3"

	createOpts := endpoints.CreateOpts{
		Availability: gophercloud.AvailabilityPublic,
		Name:         "neutron",
		Region:       "RegionOne",
		URL:          "https://localhost:9696",
		ServiceID:    serviceID,
	}

	endpoint, err := endpoints.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}


Example to Update an Endpoint

	endpointID := "ad59deeec5154d1fa0dcff518596f499"

	updateOpts := endpoints.UpdateOpts{
		Region: "RegionTwo",
	}

	endpoint, err := endpoints.Update(identityClient, endpointID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete an Endpoint

	endpointID := "ad59deeec5154d1fa0dcff518596f499"
	err := endpoints.Delete(identityClient, endpointID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package endpoints
