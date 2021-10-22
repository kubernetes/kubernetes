/*
Package usage provides information and interaction with the
SimpleTenantUsage extension for the OpenStack Compute service.

Due to the way the API responses are formatted, it is not recommended to
query by using the AllPages convenience method. Instead, use the EachPage
method to view each result page-by-page.

This is because the usage calculations are done _per page_ and not as
an aggregated total of the entire usage set.

Example to Retrieve Usage for a Single Tenant:

	start := time.Date(2017, 01, 21, 10, 4, 20, 0, time.UTC)
	end := time.Date(2017, 01, 21, 10, 4, 20, 0, time.UTC)

	singleTenantOpts := usage.SingleTenantOpts{
		Start: &start,
		End: &end,
	}

	err := usage.SingleTenant(computeClient, tenantID, singleTenantOpts).EachPage(func(page pagination.Page) (bool, error) {
		tenantUsage, err := usage.ExtractSingleTenant(page)
		if err != nil {
			return false, err
		}

		fmt.Printf("%+v\n", tenantUsage)

		return true, nil
	})

	if err != nil {
		panic(err)
	}

Example to Retrieve Usage for All Tenants:

	allTenantsOpts := usage.AllTenantsOpts{
		Detailed: true,
	}

	err := usage.AllTenants(computeClient, allTenantsOpts).EachPage(func(page pagination.Page) (bool, error) {
		allTenantsUsage, err := usage.ExtractAllTenants(page)
		if err != nil {
			return false, err
		}

		fmt.Printf("%+v\n", allTenantsUsage)

		return true, nil
	})

	if err != nil {
		panic(err)
	}

*/
package usage
