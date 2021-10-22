/*
Package profiletypes lists all profile types and shows details for a profile type from the OpenStack
Clustering Service.

Example to List ProfileType

	err = profiletypes.List(serviceClient).EachPage(func(page pagination.Page) (bool, error) {
		profileTypes, err := profiletypes.ExtractProfileTypes(page)
		if err != nil {
			return false, err
		}

		for _, profileType := range profileTypes {
			fmt.Println("%+v\n", profileType)
		}
		return true, nil
	})

Example to Get a ProfileType

	profileTypeName := "os.nova.server"
	profileType, err := profiletypes.Get(clusteringClient, profileTypeName).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", profileType)

Example of list operations supported by a profile type
	serviceClient.Microversion = "1.5"

	profileTypeName := "os.nova.server-1.0"
	allPages, err := profiletypes.ListOps(serviceClient, profileTypeName).AllPages()
	if err != nil {
		panic(err)
	}

	ops, err := profiletypes.ExtractOps(allPages)
	if err != nil {
		panic(err)
	}

	for _, op := range ops {
		fmt.Printf("%+v\n", op)
	}

*/
package profiletypes
