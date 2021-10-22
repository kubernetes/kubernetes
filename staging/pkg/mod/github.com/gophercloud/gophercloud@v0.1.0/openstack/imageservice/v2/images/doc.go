/*
Package images enables management and retrieval of images from the OpenStack
Image Service.

Example to List Images

	images.ListOpts{
		Owner: "a7509e1ae65945fda83f3e52c6296017",
	}

	allPages, err := images.List(imagesClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allImages, err := images.ExtractImages(allPages)
	if err != nil {
		panic(err)
	}

	for _, image := range allImages {
		fmt.Printf("%+v\n", image)
	}

Example to Create an Image

	createOpts := images.CreateOpts{
		Name:       "image_name",
		Visibility: images.ImageVisibilityPrivate,
	}

	image, err := images.Create(imageClient, createOpts)
	if err != nil {
		panic(err)
	}

Example to Update an Image

	imageID := "1bea47ed-f6a9-463b-b423-14b9cca9ad27"

	updateOpts := images.UpdateOpts{
		images.ReplaceImageName{
			NewName: "new_name",
		},
	}

	image, err := images.Update(imageClient, imageID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete an Image

	imageID := "1bea47ed-f6a9-463b-b423-14b9cca9ad27"
	err := images.Delete(imageClient, imageID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package images
