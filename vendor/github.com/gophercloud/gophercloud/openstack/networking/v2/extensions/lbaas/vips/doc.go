/*
Package vips provides information and interaction with the Virtual IPs of the
Load Balancing as a Service extension for the OpenStack Networking service.

Example to List Virtual IPs

	listOpts := vips.ListOpts{
		SubnetID: "d9bd223b-f1a9-4f98-953b-df977b0f902d",
	}

	allPages, err := vips.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allVIPs, err := vips.ExtractVIPs(allPages)
	if err != nil {
		panic(err)
	}

	for _, vip := range allVIPs {
		fmt.Printf("%+v\n", vip)
	}

Example to Create a Virtual IP

	createOpts := vips.CreateOpts{
		Protocol:     "HTTP",
		Name:         "NewVip",
		AdminStateUp: gophercloud.Enabled,
		SubnetID:     "8032909d-47a1-4715-90af-5153ffe39861",
		PoolID:       "61b1f87a-7a21-4ad3-9dda-7f81d249944f",
		ProtocolPort: 80,
		Persistence:  &vips.SessionPersistence{Type: "SOURCE_IP"},
	}

	vip, err := vips.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Virtual IP

	vipID := "93f1bad4-0423-40a8-afac-3fc541839912"

	i1000 := 1000
	updateOpts := vips.UpdateOpts{
		ConnLimit:   &i1000,
		Persistence: &vips.SessionPersistence{Type: "SOURCE_IP"},
	}

	vip, err := vips.Update(networkClient, vipID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Virtual IP

	vipID := "93f1bad4-0423-40a8-afac-3fc541839912"
	err := vips.Delete(networkClient, vipID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package vips
