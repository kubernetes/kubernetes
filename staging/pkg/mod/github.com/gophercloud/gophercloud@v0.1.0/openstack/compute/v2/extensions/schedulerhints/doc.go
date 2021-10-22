/*
Package schedulerhints extends the server create request with the ability to
specify additional parameters which determine where the server will be
created in the OpenStack cloud.

Example to Add a Server to a Server Group

	schedulerHints := schedulerhints.SchedulerHints{
		Group: "servergroup-uuid",
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		ImageRef:  "image-uuid",
		FlavorRef: "flavor-uuid",
	}

	createOpts := schedulerhints.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		SchedulerHints:    schedulerHints,
	}

	server, err := servers.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Place Server B on a Different Host than Server A

	schedulerHints := schedulerhints.SchedulerHints{
		DifferentHost: []string{
			"server-a-uuid",
		}
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_b",
		ImageRef:  "image-uuid",
		FlavorRef: "flavor-uuid",
	}

	createOpts := schedulerhints.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		SchedulerHints:    schedulerHints,
	}

	server, err := servers.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Place Server B on the Same Host as Server A

	schedulerHints := schedulerhints.SchedulerHints{
		SameHost: []string{
			"server-a-uuid",
		}
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_b",
		ImageRef:  "image-uuid",
		FlavorRef: "flavor-uuid",
	}

	createOpts := schedulerhints.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		SchedulerHints:    schedulerHints,
	}

	server, err := servers.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package schedulerhints
