/*
Package containers manages and retrieves containers in the OpenStack Key Manager
Service.

Example to List Containers

	allPages, err := containers.List(client, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allContainers, err := containers.ExtractContainers(allPages)
	if err != nil {
		panic(err)
	}

	for _, v := range allContainers {
		fmt.Printf("%v\n", v)
	}

Example to Create a Container

	createOpts := containers.CreateOpts{
		Type: containers.GenericContainer,
		Name: "mycontainer",
		SecretRefs: []containers.SecretRef{
			{
				Name: secret.Name,
				SecretRef: secret.SecretRef,
			},
		},
	}

	container, err := containers.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", container)

Example to Delete a Container

	err := containers.Delete(client, containerID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List Consumers of a Container

	allPages, err := containers.ListConsumers(client, containerID, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allConsumers, err := containers.ExtractConsumers(allPages)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", allConsumers)

Example to Create a Consumer of a Container

	createOpts := containers.CreateConsumerOpts{
		Name: "jdoe",
		URL:  "http://example.com",
	}

	container, err := containers.CreateConsumer(client, containerID, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Consumer of a Container

	deleteOpts := containers.DeleteConsumerOpts{
		Name: "jdoe",
		URL:  "http://example.com",
	}

	container, err := containers.DeleteConsumer(client, containerID, deleteOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package containers
