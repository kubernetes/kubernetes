/*
Package objects contains functionality for working with Object Storage
object resources. An object is a resource that represents and contains data
- such as documents, images, and so on. You can also store custom metadata
with an object.

Note: When referencing the Object Storage API docs, some of the API actions
are listed under "containers" rather than "objects". This was an intentional
design in Gophercloud to make some object actions feel more natural.

Example to List Objects

	containerName := "my_container"

	listOpts := objects.ListOpts{
		Full: true,
	}

	allPages, err := objects.List(objectStorageClient, containerName, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allObjects, err := objects.ExtractInfo(allPages)
	if err != nil {
		panic(err)
	}

	for _, object := range allObjects {
		fmt.Printf("%+v\n", object)
	}

Example to List Object Names

	containerName := "my_container"

	listOpts := objects.ListOpts{
		Full: false,
	}

	allPages, err := objects.List(objectStorageClient, containerName, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allObjects, err := objects.ExtractNames(allPages)
	if err != nil {
		panic(err)
	}

	for _, object := range allObjects {
		fmt.Printf("%+v\n", object)
	}

Example to Create an Object

	content := "some object content"
	objectName := "my_object"
	containerName := "my_container"

	createOpts := objects.CreateOpts{
		ContentType: "text/plain"
		Content:     strings.NewReader(content),
	}

	object, err := objects.Create(objectStorageClient, containerName, objectName, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Copy an Object

	objectName := "my_object"
	containerName := "my_container"

	copyOpts := objects.CopyOpts{
		Destination: "/newContainer/newObject",
	}

	object, err := objects.Copy(objectStorageClient, containerName, objectName, copyOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete an Object

	objectName := "my_object"
	containerName := "my_container"

	object, err := objects.Delete(objectStorageClient, containerName, objectName).Extract()
	if err != nil {
		panic(err)
	}

Example to Download an Object's Data

	objectName := "my_object"
	containerName := "my_container"

	object := objects.Download(objectStorageClient, containerName, objectName, nil)
	content, err := object.ExtractContent()
	if err != nil {
		panic(err)
	}
*/
package objects
