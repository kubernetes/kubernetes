/*
Package containers contains functionality for working with Object Storage
container resources. A container serves as a logical namespace for objects
that are placed inside it - an object with the same name in two different
containers represents two different objects.

In addition to containing objects, you can also use the container to control
access to objects by using an access control list (ACL).

Example to List Containers

	listOpts := containers.ListOpts{
		Full: true,
	}

	allPages, err := containers.List(objectStorageClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allContainers, err := containers.ExtractInfo(allPages)
	if err != nil {
		panic(err)
	}

	for _, container := range allContainers {
		fmt.Printf("%+v\n", container)
	}

Example to List Only Container Names

	listOpts := containers.ListOpts{
		Full: false,
	}

	allPages, err := containers.List(objectStorageClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allContainers, err := containers.ExtractNames(allPages)
	if err != nil {
		panic(err)
	}

	for _, container := range allContainers {
		fmt.Printf("%+v\n", container)
	}

Example to Create a Container

	createOpts := containers.CreateOpts{
		ContentType: "application/json",
		Metadata: map[string]string{
			"foo": "bar",
		},
	}

	container, err := containers.Create(objectStorageClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Container

	containerName := "my_container"

	updateOpts := containers.UpdateOpts{
		Metadata: map[string]string{
			"bar": "baz",
		},
	}

	container, err := containers.Update(objectStorageClient, containerName, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Container

	containerName := "my_container"

	container, err := containers.Delete(objectStorageClient, containerName).Extract()
	if err != nil {
		panic(err)
	}
*/
package containers
