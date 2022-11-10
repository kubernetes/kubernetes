/*
Package extensions provides information and interaction with the different
extensions available for an OpenStack service.

The purpose of OpenStack API extensions is to:

- Introduce new features in the API without requiring a version change.
- Introduce vendor-specific niche functionality.
- Act as a proving ground for experimental functionalities that might be
included in a future version of the API.

Extensions usually have tags that prevent conflicts with other extensions that
define attributes or resources with the same names, and with core resources and
attributes. Because an extension might not be supported by all plug-ins, its
availability varies with deployments and the specific plug-in.

The results of this package vary depending on the type of Service Client used.
In the following examples, note how the only difference is the creation of the
Service Client.

Example of Retrieving Compute Extensions

	ao, err := openstack.AuthOptionsFromEnv()
	provider, err := openstack.AuthenticatedClient(ao)
	computeClient, err := openstack.NewComputeV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})

	allPages, err := extensions.List(computeClient).Allpages()
	allExtensions, err := extensions.ExtractExtensions(allPages)

	for _, extension := range allExtensions{
		fmt.Println("%+v\n", extension)
	}


Example of Retrieving Network Extensions

	ao, err := openstack.AuthOptionsFromEnv()
	provider, err := openstack.AuthenticatedClient(ao)
	networkClient, err := openstack.NewNetworkV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})

	allPages, err := extensions.List(networkClient).Allpages()
	allExtensions, err := extensions.ExtractExtensions(allPages)

	for _, extension := range allExtensions{
		fmt.Println("%+v\n", extension)
	}
*/
package extensions
