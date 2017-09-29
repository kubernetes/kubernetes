/*
Package ports contains functionality for working with Neutron port resources.

A port represents a virtual switch port on a logical network switch. Virtual
instances attach their interfaces into ports. The logical port also defines
the MAC address and the IP address(es) to be assigned to the interfaces
plugged into them. When IP addresses are associated to a port, this also
implies the port is associated with a subnet, as the IP address was taken
from the allocation pool for a specific subnet.

Example to List Ports

	listOpts := ports.ListOpts{
		DeviceID: "b0b89efe-82f8-461d-958b-adbf80f50c7d",
	}

	allPages, err := ports.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allPorts, err := ports.ExtractPorts(allPages)
	if err != nil {
		panic(err)
	}

	for _, port := range allPorts {
		fmt.Printf("%+v\n", port)
	}

Example to Create a Port

	createOtps := ports.CreateOpts{
		Name:         "private-port",
		AdminStateUp: &asu,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
		SecurityGroups: &[]string{"foo"},
		AllowedAddressPairs: []ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}

	port, err := ports.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Port

	portID := "c34bae2b-7641-49b6-bf6d-d8e473620ed8"

	updateOpts := ports.UpdateOpts{
		Name:           "new_name",
		SecurityGroups: &[]string{},
	}

	port, err := ports.Update(networkClient, portID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Port

	portID := "c34bae2b-7641-49b6-bf6d-d8e473620ed8"
	err := ports.Delete(networkClient, portID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package ports
