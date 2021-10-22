/*
Package firewalls allows management and retrieval of firewalls from the
OpenStack Networking Service.

Example to List Firewalls

	listOpts := firewalls.ListOpts{
		TenantID: "tenant-id",
	}

	allPages, err := firewalls.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allFirewalls, err := firewalls.ExtractFirewalls(allPages)
	if err != nil {
		panic(err)
	}

	for _, fw := range allFirewalls {
		fmt.Printf("%+v\n", fw)
	}

Example to Create a Firewall

	createOpts := firewalls.CreateOpts{
		Name:        "firewall_1",
		Description: "A firewall",
		PolicyID:    "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
		AdminStateUp: gophercloud.Enabled,
	}

	firewall, err := firewalls.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Firewall

	firewallID := "a6917946-38ab-4ffd-a55a-26c0980ce5ee"

	updateOpts := firewalls.UpdateOpts{
		AdminStateUp: gophercloud.Disabled,
	}

	firewall, err := firewalls.Update(networkClient, firewallID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Firewall

	firewallID := "a6917946-38ab-4ffd-a55a-26c0980ce5ee"
	err := firewalls.Delete(networkClient, firewallID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package firewalls
