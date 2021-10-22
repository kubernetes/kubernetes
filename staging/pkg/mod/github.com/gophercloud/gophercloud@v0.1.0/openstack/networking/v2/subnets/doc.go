/*
Package subnets contains functionality for working with Neutron subnet
resources. A subnet represents an IP address block that can be used to
assign IP addresses to virtual instances. Each subnet must have a CIDR and
must be associated with a network. IPs can either be selected from the whole
subnet CIDR or from allocation pools specified by the user.

A subnet can also have a gateway, a list of DNS name servers, and host routes.
This information is pushed to instances whose interfaces are associated with
the subnet.

Example to List Subnets

	listOpts := subnets.ListOpts{
		IPVersion: 4,
	}

	allPages, err := subnets.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allSubnets, err := subnets.ExtractSubnets(allPages)
	if err != nil {
		panic(err)
	}

	for _, subnet := range allSubnets {
		fmt.Printf("%+v\n", subnet)
	}

Example to Create a Subnet With Specified Gateway

	var gatewayIP = "192.168.199.1"
	createOpts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		IPVersion: 4,
		CIDR:      "192.168.199.0/24",
		GatewayIP: &gatewayIP,
		AllocationPools: []subnets.AllocationPool{
		  {
		    Start: "192.168.199.2",
		    End:   "192.168.199.254",
		  },
		},
		DNSNameservers: []string{"foo"},
	}

	subnet, err := subnets.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Create a Subnet With No Gateway

	var noGateway = ""

	createOpts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a23",
		IPVersion: 4,
		CIDR:      "192.168.1.0/24",
		GatewayIP: &noGateway,
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "192.168.1.2",
				End:   "192.168.1.254",
			},
		},
		DNSNameservers: []string{},
	}

	subnet, err := subnets.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Create a Subnet With a Default Gateway

	createOpts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a23",
		IPVersion: 4,
		CIDR:      "192.168.1.0/24",
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "192.168.1.2",
				End:   "192.168.1.254",
			},
		},
		DNSNameservers: []string{},
	}

	subnet, err := subnets.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Subnet

	subnetID := "db77d064-e34f-4d06-b060-f21e28a61c23"
	dnsNameservers := []string{"8.8.8.8"}
	name := "new_name"

	updateOpts := subnets.UpdateOpts{
		Name:           &name,
		DNSNameservers: &dnsNameservers,
	}

	subnet, err := subnets.Update(networkClient, subnetID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Remove a Gateway From a Subnet

	var noGateway = ""
	subnetID := "db77d064-e34f-4d06-b060-f21e28a61c23"

	updateOpts := subnets.UpdateOpts{
		GatewayIP: &noGateway,
	}

	subnet, err := subnets.Update(networkClient, subnetID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Subnet

	subnetID := "db77d064-e34f-4d06-b060-f21e28a61c23"
	err := subnets.Delete(networkClient, subnetID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package subnets
