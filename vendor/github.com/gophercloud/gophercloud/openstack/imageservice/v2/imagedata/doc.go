/*
Package imagedata enables management of image data.

Example to Upload Image Data

	imageID := "da3b75d9-3f4a-40e7-8a2c-bfab23927dea"

	imageData, err := os.Open("/path/to/image/file")
	if err != nil {
		panic(err)
	}
	defer imageData.Close()

	err = imagedata.Upload(imageClient, imageID, imageData).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Download Image Data

	imageID := "da3b75d9-3f4a-40e7-8a2c-bfab23927dea"

	image, err := imagedata.Download(imageClient, imageID).Extract()
	if err != nil {
		panic(err)
	}

	imageData, err := ioutil.ReadAll(image)
	if err != nil {
		panic(err)
	}
*/
package imagedata
