/*
Package accounts contains functionality for working with Object Storage
account resources. An account is the top-level resource the object storage
hierarchy: containers belong to accounts, objects belong to containers.

Another way of thinking of an account is like a namespace for all your
resources. It is synonymous with a project or tenant in other OpenStack
services.

Example to Get an Account

	account, err := accounts.Get(objectStorageClient, nil).Extract()
	fmt.Printf("%+v\n", account)

Example to Update an Account

	metadata := map[string]string{
		"some": "metadata",
	}

	updateOpts := accounts.UpdateOpts{
		Metadata: metadata,
	}

	updateResult, err := accounts.Update(objectStorageClient, updateOpts).Extract()
	fmt.Printf("%+v\n", updateResult)

*/
package accounts
