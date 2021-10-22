/*
Package ikepolicies allows management and retrieval of IKE policies in the
OpenStack Networking Service.


Example to Create an IKE policy

	createOpts := ikepolicies.CreateOpts{
		Name:                "ikepolicy1",
		Description:         "Description of ikepolicy1",
		EncryptionAlgorithm: ikepolicies.EncryptionAlgorithm3DES,
		PFS:                 ikepolicies.PFSGroup5,
	}

	policy, err := ikepolicies.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Show the details of a specific IKE policy by ID

	policy, err := ikepolicies.Get(client, "f2b08c1e-aa81-4668-8ae1-1401bcb0576c").Extract()
	if err != nil {
		panic(err)
	}


Example to Delete a Policy

	err := ikepolicies.Delete(client, "5291b189-fd84-46e5-84bd-78f40c05d69c").ExtractErr()
	if err != nil {
		panic(err)

Example to Update an IKE policy

	name := "updatedname"
	description := "updated policy"
	updateOpts := ikepolicies.UpdateOpts{
		Name:        &name,
		Description: &description,
		Lifetime: &ikepolicies.LifetimeUpdateOpts{
			Value: 7000,
		},
	}
	updatedPolicy, err := ikepolicies.Update(client, "5c561d9d-eaea-45f6-ae3e-08d1a7080828", updateOpts).Extract()
	if err != nil {
		panic(err)
	}


Example to List IKE policies

	allPages, err := ikepolicies.List(client, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allPolicies, err := ikepolicies.ExtractPolicies(allPages)
	if err != nil {
		panic(err)
	}

*/
package ikepolicies
