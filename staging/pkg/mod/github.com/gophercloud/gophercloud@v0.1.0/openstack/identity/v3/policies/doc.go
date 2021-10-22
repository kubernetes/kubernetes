/*
Package policies provides information and interaction with the policies API
resource for the OpenStack Identity service.

Example to List Policies

	listOpts := policies.ListOpts{
		Type: "application/json",
	}

	allPages, err := policies.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allPolicies, err := policies.ExtractPolicies(allPages)
	if err != nil {
		panic(err)
	}

	for _, policy := range allPolicies {
		fmt.Printf("%+v\n", policy)
	}

Example to Create a Policy

	createOpts := policies.CreateOpts{
		Type: "application/json",
		Blob: []byte("{'foobar_user': 'role:compute-user'}"),
		Extra: map[string]interface{}{
			"description": "policy for foobar_user",
		},
	}

	policy, err := policies.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Get a Policy

	policyID := "0fe36e73809d46aeae6705c39077b1b3"
	policy, err := policies.Get(identityClient, policyID).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", policy)

Example to Update a Policy

	policyID := "0fe36e73809d46aeae6705c39077b1b3"

	updateOpts := policies.UpdateOpts{
		Type: "application/json",
		Blob: []byte("{'foobar_user': 'role:compute-user'}"),
		Extra: map[string]interface{}{
			"description": "policy for foobar_user",
		},
	}

	policy, err := policies.Update(identityClient, policyID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", policy)

Example to Delete a Policy

	policyID := "0fe36e73809d46aeae6705c39077b1b3"
	err := policies.Delete(identityClient, policyID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package policies
