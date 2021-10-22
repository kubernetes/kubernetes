/*
Package services returns information about the compute services in the OpenStack
cloud.

Example of Retrieving list of all services

	allPages, err := services.List(computeClient).AllPages()
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
*/

package services
