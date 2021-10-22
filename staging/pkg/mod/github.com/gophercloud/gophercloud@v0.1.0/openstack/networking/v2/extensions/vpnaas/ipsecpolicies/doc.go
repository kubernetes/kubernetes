/*
Package ipsecpolicies allows management and retrieval of IPSec Policies in the
OpenStack Networking Service.

Example to Create a Policy

	createOpts := ipsecpolicies.CreateOpts{
		Name:        "IPSecPolicy_1",
	}

	policy, err := policies.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Policy

	err := ipsecpolicies.Delete(client, "5291b189-fd84-46e5-84bd-78f40c05d69c").ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Show the details of a specific IPSec policy by ID

	policy, err := ipsecpolicies.Get(client, "f2b08c1e-aa81-4668-8ae1-1401bcb0576c").Extract()
	if err != nil {
		panic(err)
	}

Example to Update an IPSec policy

	name := "updatedname"
	description := "updated policy"
	updateOpts := ipsecpolicies.UpdateOpts{
		Name:        &name,
		Description: &description,
	}
	updatedPolicy, err := ipsecpolicies.Update(client, "5c561d9d-eaea-45f6-ae3e-08d1a7080828", updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to List IPSec policies

	allPages, err := ipsecpolicies.List(client, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allPolicies, err := ipsecpolicies.ExtractPolicies(allPages)
	if err != nil {
		panic(err)
	}

*/
package ipsecpolicies
