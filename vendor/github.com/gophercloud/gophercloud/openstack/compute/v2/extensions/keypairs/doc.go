/*
Package keypairs provides the ability to manage key pairs as well as create
servers with a specified key pair.

Example to List Key Pairs

	allPages, err := keypairs.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allKeyPairs, err := keypairs.ExtractKeyPairs(allPages)
	if err != nil {
		panic(err)
	}

	for _, kp := range allKeyPairs {
		fmt.Printf("%+v\n", kp)
	}

Example to Create a Key Pair

	createOpts := keypairs.CreateOpts{
		Name: "keypair-name",
	}

	keypair, err := keypairs.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v", keypair)

Example to Import a Key Pair

	createOpts := keypairs.CreateOpts{
		Name:      "keypair-name",
		PublicKey: "public-key",
	}

	keypair, err := keypairs.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Key Pair

	err := keypairs.Delete(computeClient, "keypair-name").ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Create a Server With a Key Pair

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		ImageRef:  "image-uuid",
		FlavorRef: "flavor-uuid",
	}

	createOpts := keypairs.CreateOpts{
		CreateOptsBuilder: serverCreateOpts,
		KeyName:           "keypair-name",
	}

	server, err := servers.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package keypairs
