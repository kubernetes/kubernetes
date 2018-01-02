/*
Package routerinsertion implements the fwaasrouterinsertion Firewall extension.
It is used to manage the router information of a firewall.

Example to List Firewalls with Router Information

	type FirewallsWithRouters struct {
		firewalls.Firewall
		routerinsertion.FirewallExt
	}

	var allFirewalls []FirewallsWithRouters

	allPages, err := firewalls.List(networkClient, nil).AllPages()
	if err != nil {
		panic(err)
	}

	err = firewalls.ExtractFirewallsInto(allPages, &allFirewalls)
	if err != nil {
		panic(err)
	}

	for _, fw := range allFirewalls {
		fmt.Printf("%+v\n", fw)
	}

Example to Create a Firewall with a Router

	firewallCreateOpts := firewalls.CreateOpts{
		Name:     "firewall_1",
		PolicyID: "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}

	createOpts := routerinsertion.CreateOptsExt{
		CreateOptsBuilder: firewallCreateOpts,
		RouterIDs: []string{
			"8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8",
		},
	}

	firewall, err := firewalls.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Firewall with a Router

	firewallID := "a6917946-38ab-4ffd-a55a-26c0980ce5ee"

	firewallUpdateOpts := firewalls.UpdateOpts{
		Description: "updated firewall",
		PolicyID:    "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}

	updateOpts := routerinsertion.UpdateOptsExt{
		UpdateOptsBuilder: firewallUpdateOpts,
		RouterIDs: []string{
			"8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8",
		},
	}

	firewall, err := firewalls.Update(networkClient, firewallID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package routerinsertion
