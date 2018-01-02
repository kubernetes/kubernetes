/*
Package services provides information and interaction with the services API
resource for the OpenStack Identity service.

Example to List Services

	listOpts := services.ListOpts{
		ServiceType: "compute",
	}

	allPages, err := services.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allServices, err := services.ExtractServices(allPages)
	if err != nil {
		panic(err)
	}

	for _, service := range allServices {
		fmt.Printf("%+v\n", service)
	}

Example to Create a Service

	createOpts := services.CreateOpts{
		Type: "compute",
		Extra: map[string]interface{}{
			"name": "compute-service",
			"description": "Compute Service",
		},
	}

	service, err := services.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Service

	serviceID :=  "3c7bbe9a6ecb453ca1789586291380ed"

	var iFalse bool = false
	updateOpts := services.UpdateOpts{
		Enabled: &iFalse,
		Extra: map[string]interface{}{
			"description": "Disabled Compute Service"
		},
	}

	service, err := services.Update(identityClient, serviceID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Service

	serviceID := "3c7bbe9a6ecb453ca1789586291380ed"
	err := services.Delete(identityClient, serviceID).ExtractErr()
	if err != nil {
		panic(err)
	}

*/
package services
