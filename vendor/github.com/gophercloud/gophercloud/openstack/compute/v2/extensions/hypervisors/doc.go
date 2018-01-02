/*
Package hypervisors returns details about the hypervisors in the OpenStack
cloud.

Example of Retrieving Details of All Hypervisors

	allPages, err := hypervisors.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allHypervisors, err := hypervisors.ExtractHypervisors(allPages)
	if err != nil {
		panic(err)
	}

	for _, hypervisor := range allHypervisors {
		fmt.Printf("%+v\n", hypervisor)
	}
*/
package hypervisors
