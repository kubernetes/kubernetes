/*
Package policies provides information and interaction with the policies through
the OpenStack Clustering service.

Example to List Policies

	listOpts := policies.ListOpts{
		Limit: 2,
	}

	allPages, err := policies.List(clusteringClient, listOpts).AllPages()
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

	opts := policies.CreateOpts{
		Name: "new_policy",
		Spec: policies.Spec{
			Description: "new policy description",
			Properties: map[string]interface{}{
				"hooks": map[string]interface{}{
					"type": "zaqar",
					"params": map[string]interface{}{
						"queue": "my_zaqar_queue",
					},
					"timeout": 10,
				},
			},
			Type:    "senlin.policy.deletion",
			Version: "1.1",
		},
	}

	createdPolicy, err := policies.Create(client, opts).Extract()
	if err != nil {
		panic(err)
	}

Example to Get a Policy

    policyName := "get_policy"
    policyDetail, err := policies.Get(clusteringClient, policyName).Extract()
    if err != nil {
        panic(err)
    }

    fmt.Printf("%+v\n", policyDetail)

Example to Update a Policy

	opts := policies.UpdateOpts{
		Name: "update_policy",
	}

	updatePolicy, err := policies.Update(client, opts).Extract()
	if err != nil {
		panic(err)
	}

Example to Validate a Policy

	opts := policies.ValidateOpts{
		Spec: policies.Spec{
			Description: "new policy description",
			Properties: map[string]interface{}{
				"hooks": map[string]interface{}{
					"type": "zaqar",
					"params": map[string]interface{}{
						"queue": "my_zaqar_queue",
					},
					"timeout": 10,
				},
			},
			Type:    "senlin.policy.deletion",
			Version: "1.1",
		},
	}

	validatePolicy, err := policies.Validate(client, opts).Extract()
	if err != nil {
		panic(err)
	}
*/
package policies
