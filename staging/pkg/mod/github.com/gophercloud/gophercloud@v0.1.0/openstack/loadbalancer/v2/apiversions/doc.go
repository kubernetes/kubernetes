/*
Package apiversions provides information and interaction with the different
API versions for the OpenStack Load Balancer service. This functionality is not
restricted to this particular version.

Example to List API Versions

	allPages, err := apiversions.List(loadbalancerClient).AllPages()
	if err != nil {
		panic(err)
	}

	allVersions, err := apiversions.ExtractAPIVersions(allPages)
	if err != nil {
		panic(err)
	}

	for _, version := range allVersions {
		fmt.Printf("%+v\n", version)
	}
*/
package apiversions
