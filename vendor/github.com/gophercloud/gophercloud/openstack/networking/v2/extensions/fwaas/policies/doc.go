/*
Package policies allows management and retrieval of Firewall Policies in the
OpenStack Networking Service.

Example to List Policies

	listOpts := policies.ListOpts{
		TenantID: "966b3c7d36a24facaf20b7e458bf2192",
	}

	allPages, err := policies.List(networkClient, listOpts).AllPages()
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
		Name:        "policy_1",
		Description: "A policy",
		Rules: []string{
			"98a58c87-76be-ae7c-a74e-b77fffb88d95",
			"7c4f087a-ed46-4ea8-8040-11ca460a61c0",
		}
	}

	policy, err := policies.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Policy

	policyID := "38aee955-6283-4279-b091-8b9c828000ec"

	updateOpts := policies.UpdateOpts{
		Description: "New Description",
	}

	policy, err := policies.Update(networkClient, policyID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Policy

	policyID := "38aee955-6283-4279-b091-8b9c828000ec"
	err := policies.Delete(networkClient, policyID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Add a Rule to a Policy

	policyID := "38aee955-6283-4279-b091-8b9c828000ec"
	ruleOpts := policies.InsertRuleOpts{
		ID: "98a58c87-76be-ae7c-a74e-b77fffb88d95",
	}

	policy, err := policies.AddRule(networkClient, policyID, ruleOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Rule from a Policy

	policyID := "38aee955-6283-4279-b091-8b9c828000ec"
	ruleID := "98a58c87-76be-ae7c-a74e-b77fffb88d95",

	policy, err := policies.RemoveRule(networkClient, policyID, ruleID).Extract()
	if err != nil {
		panic(err)
	}
*/
package policies
