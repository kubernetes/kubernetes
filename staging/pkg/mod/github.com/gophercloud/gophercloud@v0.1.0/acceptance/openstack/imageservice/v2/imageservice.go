// Package v2 contains common functions for creating imageservice resources
// for use in acceptance tests. See the `*_test.go` files for example usages.
package v2

import (
	"io"
	"net/http"
	"os"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/imagedata"
	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/imageimport"
	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/images"
	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/tasks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateEmptyImage will create an image, but with no actual image data.
// An error will be returned if an image was unable to be created.
func CreateEmptyImage(t *testing.T, client *gophercloud.ServiceClient) (*images.Image, error) {
	var image *images.Image

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create image: %s", name)

	protected := false
	visibility := images.ImageVisibilityPrivate
	createOpts := &images.CreateOpts{
		Name:            name,
		ContainerFormat: "bare",
		DiskFormat:      "qcow2",
		MinDisk:         0,
		MinRAM:          0,
		Protected:       &protected,
		Visibility:      &visibility,
		Properties: map[string]string{
			"architecture": "x86_64",
		},
		Tags: []string{"foo", "bar", "baz"},
	}

	image, err := images.Create(client, createOpts).Extract()
	if err != nil {
		return image, err
	}

	newImage, err := images.Get(client, image.ID).Extract()
	if err != nil {
		return image, err
	}

	t.Logf("Created image %s: %#v", name, newImage)

	th.CheckEquals(t, newImage.Name, name)
	th.CheckEquals(t, newImage.Properties["architecture"], "x86_64")
	return newImage, nil
}

// DeleteImage deletes an image.
// A fatal error will occur if the image failed to delete. This works best when
// used as a deferred function.
func DeleteImage(t *testing.T, client *gophercloud.ServiceClient, image *images.Image) {
	err := images.Delete(client, image.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete image %s: %v", image.ID, err)
	}

	t.Logf("Deleted image: %s", image.ID)
}

// ImportImageURL contains an URL of a test image that can be imported.
const ImportImageURL = "http://download.cirros-cloud.net/0.4.0/cirros-0.4.0-x86_64-disk.img"

// CreateTask will create a task to import the CirrOS image.
// An error will be returned if a task couldn't be created.
func CreateTask(t *testing.T, client *gophercloud.ServiceClient, imageURL string) (*tasks.Task, error) {
	t.Logf("Attempting to create an Imageservice import task with image: %s", imageURL)
	opts := tasks.CreateOpts{
		Type: "import",
		Input: map[string]interface{}{
			"image_properties": map[string]interface{}{
				"container_format": "bare",
				"disk_format":      "raw",
			},
			"import_from_format": "raw",
			"import_from":        imageURL,
		},
	}
	task, err := tasks.Create(client, opts).Extract()
	if err != nil {
		return nil, err
	}

	newTask, err := tasks.Get(client, task.ID).Extract()
	if err != nil {
		return nil, err
	}

	return newTask, nil
}

// GetImportInfo will retrieve Import API information.
func GetImportInfo(t *testing.T, client *gophercloud.ServiceClient) (*imageimport.ImportInfo, error) {
	t.Log("Attempting to get the Imageservice Import API information")
	importInfo, err := imageimport.Get(client).Extract()
	if err != nil {
		return nil, err
	}

	return importInfo, nil
}

// StageImage will stage local image file to the referenced remote queued image.
func StageImage(t *testing.T, client *gophercloud.ServiceClient, filepath, imageID string) error {
	imageData, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer imageData.Close()

	return imagedata.Stage(client, imageID, imageData).ExtractErr()
}

// DownloadImageFileFromURL will download an image from the specified URL and
// place it into the specified path.
func DownloadImageFileFromURL(t *testing.T, url, filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	t.Logf("Attempting to download image from %s", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	size, err := io.Copy(file, resp.Body)
	if err != nil {
		return err
	}

	t.Logf("Downloaded image with size of %d bytes in %s", size, filepath)
	return nil
}

// DeleteImageFile will delete local image file.
func DeleteImageFile(t *testing.T, filepath string) {
	err := os.Remove(filepath)
	if err != nil {
		t.Fatalf("Unable to delete image file %s", filepath)
	}

	t.Logf("Successfully deleted image file %s", filepath)
}

// ImportImage will import image data from the remote source to the Imageservice.
func ImportImage(t *testing.T, client *gophercloud.ServiceClient, imageID string) error {
	importOpts := imageimport.CreateOpts{
		Name: imageimport.WebDownloadMethod,
		URI:  ImportImageURL,
	}

	t.Logf("Attempting to import image data for %s from %s", imageID, importOpts.URI)
	return imageimport.Create(client, imageID, importOpts).ExtractErr()
}
