/*
Package rbacpolicies contains functionality for working with Neutron RBAC Policies.
Role-Based Access Control (RBAC) policy framework enables both operators
and users to grant access to resources for specific projects.

Sharing an object with a specific project is accomplished by creating a
policy entry that permits the target project the access_as_shared action
on that object.

To make a network available as an external network for specific projects
rather than all projects, use the access_as_external action.
If a network is marked as external during creation, it now implicitly creates
a wildcard RBAC policy granting everyone access to preserve previous behavior
before this feature was added.

Example to Create a RBAC Policy

	createOpts := rbacpolicies.CreateOpts{
		Action:       rbacpolicies.ActionAccessShared,
		ObjectType:   "network",
                TargetTenant: "6e547a3bcfe44702889fdeff3c3520c3",
                ObjectID:     "240d22bf-bd17-4238-9758-25f72610ecdc"
	}

	rbacPolicy, err := rbacpolicies.Create(rbacClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to List RBAC Policies

	listOpts := rbacpolicies.ListOpts{
		TenantID: "a99e9b4e620e4db09a2dfb6e42a01e66",
	}

	allPages, err := rbacpolicies.List(rbacClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRBACPolicies, err := rbacpolicies.ExtractRBACPolicies(allPages)
	if err != nil {
		panic(err)
	}

	for _, rbacpolicy := range allRBACPolicies {
		fmt.Printf("%+v", rbacpolicy)
	}

Example to Delete a RBAC Policy

	rbacPolicyID := "94fe107f-da78-4d92-a9d7-5611b06dad8d"
	err := rbacpolicies.Delete(rbacClient, rbacPolicyID).ExtractErr()
	if err != nil {
	  panic(err)
	}

Example to Get RBAC Policy by ID

	rbacPolicyID := "94fe107f-da78-4d92-a9d7-5611b06dad8d"
	rbacpolicy, err := rbacpolicies.Get(rbacClient, rbacPolicyID).Extract()
	if err != nil {
	  panic(err)
	}
	fmt.Printf("%+v", rbacpolicy)

Example to Update a RBAC Policy

	rbacPolicyID := "570b0306-afb5-4d3b-ab47-458fdc16baaa"
	updateOpts := rbacpolicies.UpdateOpts{
		TargetTenant: "9d766060b6354c9e8e2da44cab0e8f38",
	}
	rbacPolicy, err := rbacpolicies.Update(rbacClient, rbacPolicyID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

*/
package rbacpolicies
