/*
Package ruletypes contains functionality for working with Neutron 'quality of service' rule-type resources.

Example: You can list rule-types in the following way:

	page, err := ruletypes.ListRuleTypes(client).AllPages()
	if err != nil {
		return
	}

	rules, err := ruletypes.ExtractRuleTypes(page)
	if err != nil {
		return
	}

	fmt.Printf("%v <- Rule Types\n", rules)

*/
package ruletypes
