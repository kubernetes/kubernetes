/*
Package l7policies provides information and interaction with L7Policies and
Rules of the LBaaS v2 extension for the OpenStack Networking service.

Example to Create a L7Policy

	createOpts := l7policies.CreateOpts{
		Name:        "redirect-example.com",
		ListenerID:  "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:      l7policies.ActionRedirectToURL,
		RedirectURL: "http://www.example.com",
	}
	l7policy, err := l7policies.Create(lbClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to List L7Policies

	listOpts := l7policies.ListOpts{
		ListenerID: "c79a4468-d788-410c-bf79-9a8ef6354852",
	}
	allPages, err := l7policies.List(lbClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}
	allL7Policies, err := l7policies.ExtractL7Policies(allPages)
	if err != nil {
		panic(err)
	}
	for _, l7policy := range allL7Policies {
		fmt.Printf("%+v\n", l7policy)
	}

Example to Get a L7Policy

	l7policy, err := l7policies.Get(lbClient, "023f2e34-7806-443b-bfae-16c324569a3d").Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a L7Policy

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	err := l7policies.Delete(lbClient, l7policyID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Update a L7Policy

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	name := "new-name"
	updateOpts := l7policies.UpdateOpts{
		Name: &name,
	}
	l7policy, err := l7policies.Update(lbClient, l7policyID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Create a Rule

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	createOpts := l7policies.CreateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images*",
	}
	rule, err := l7policies.CreateRule(lbClient, l7policyID, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to List L7 Rules

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	listOpts := l7policies.ListRulesOpts{
		RuleType: l7policies.TypePath,
	}
	allPages, err := l7policies.ListRules(lbClient, l7policyID, listOpts).AllPages()
	if err != nil {
		panic(err)
	}
	allRules, err := l7policies.ExtractRules(allPages)
	if err != nil {
		panic(err)
	}
	for _, rule := allRules {
		fmt.Printf("%+v\n", rule)
	}

Example to Get a l7 rule

	l7rule, err := l7policies.GetRule(lbClient, "023f2e34-7806-443b-bfae-16c324569a3d", "53ad8ab8-40fa-11e8-a508-00224d6b7bc1").Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a l7 rule

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	ruleID := "64dba99f-8af8-4200-8882-e32a0660f23e"
	err := l7policies.DeleteRule(lbClient, l7policyID, ruleID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Update a Rule

	l7policyID := "d67d56a6-4a86-4688-a282-f46444705c64"
	ruleID := "64dba99f-8af8-4200-8882-e32a0660f23e"
	updateOpts := l7policies.UpdateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images/special*",
	}
	rule, err := l7policies.UpdateRule(lbClient, l7policyID, ruleID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package l7policies
