/*
Package quotas contains functionality for working with Magnum Quota API.

Example to Create a Quota

	createOpts := quotas.CreateOpts{
		ProjectID: "aa5436ab58144c768ca4e9d2e9f5c3b2",
		Resource:  "Cluster",
		HardLimit: 10,
	}

	quota, err := quotas.Create(serviceClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

*/
package quotas
