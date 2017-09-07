/*
Package tenants provides information and interaction with the
tenants API resource for the OpenStack Identity service.

See http://developer.openstack.org/api-ref-identity-v2.html#identity-auth-v2
and http://developer.openstack.org/api-ref-identity-v2.html#admin-tenants
for more information.

Example to List Tenants

	listOpts := tenants.ListOpts{
		Limit: 2,
	}

	allPages, err := tenants.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allTenants, err := tenants.ExtractTenants(allPages)
	if err != nil {
		panic(err)
	}

	for _, tenant := range allTenants {
		fmt.Printf("%+v\n", tenant)
	}

Example to Create a Tenant

	createOpts := tenants.CreateOpts{
		Name:        "tenant_name",
		Description: "this is a tenant",
		Enabled:     gophercloud.Enabled,
	}

	tenant, err := tenants.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Tenant

	tenantID := "e6db6ed6277c461a853458589063b295"

	updateOpts := tenants.UpdateOpts{
		Description: "this is a new description",
		Enabled:     gophercloud.Disabled,
	}

	tenant, err := tenants.Update(identityClient, tenantID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Tenant

	tenantID := "e6db6ed6277c461a853458589063b295"

	err := tenants.Delete(identitYClient, tenantID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package tenants
