/*
Package profiles provides information and interaction with profiles through
the OpenStack Clustering service.

Example to Create a Profile

	networks := []map[string]interface{} {
		{"network": "test-network"},
	}

	props := map[string]interface{}{
		"name":            "test_gophercloud_profile",
		"flavor":          "t2.micro",
		"image":           "centos7.3-latest",
		"networks":        networks,
		"security_groups": "",
	}

	createOpts := profiles.CreateOpts {
		Name: "test_profile",
		Spec: profiles.Spec{
			Type:       "os.nova.server",
			Version:    "1.0",
			Properties: props,
		},
	}

	profile, err := profiles.Create(serviceClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Println("Profile", profile)

Example to Get a Profile

	profile, err := profiles.Get(serviceClient, "profile-name").Extract()
	if err != nil {
		panic(err)
	}

	fmt.Print("profile", profile)


Example to List Profiles

	listOpts := profiles.ListOpts{
		Limit: 2,
	}

	profiles.List(serviceClient, listOpts).EachPage(func(page pagination.Page) (bool, error) {
		allProfiles, err := profiles.ExtractProfiles(page)
		if err != nil {
			panic(err)
		}

		for _, profile := range allProfiles {
			fmt.Printf("%+v\n", profile)
		}
		return true, nil
	})

Example to Update a Profile

	updateOpts := profiles.UpdateOpts{
		Name: "new-name",
	}

	profile, err := profiles.Update(serviceClient, profileName, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Print("profile", profile)

Example to Delete a Profile

	profileID := "6dc6d336e3fc4c0a951b5698cd1236ee"
	err := profiles.Delete(serviceClient, profileID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Validate a profile

	serviceClient.Microversion = "1.2"

	validateOpts := profiles.ValidateOpts{
		Spec: profiles.Spec{
			Properties: map[string]interface{}{
				"flavor":   "t2.micro",
				"image":    "cirros-0.3.4-x86_64-uec",
				"key_name": "oskey",
				"name":     "cirros_server",
				"networks": []interface{}{
					map[string]interface{}{"network": "private"},
				},
			},
			Type:    "os.nova.server",
			Version: "1.0",
		},
	}

	profile, err := profiles.Validate(serviceClient, validateOpts).Extract()
	if err != nil {
		panic(err)
	}

*/
package profiles
