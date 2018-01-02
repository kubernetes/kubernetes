package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/images"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fakeclient "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageListSuccessfully(t)

	t.Logf("Test setup %+v\n", th.Server)

	t.Logf("Id\tName\tOwner\tChecksum\tSizeBytes")

	pager := images.List(fakeclient.ServiceClient(), images.ListOpts{Limit: 1})
	t.Logf("Pager state %v", pager)
	count, pages := 0, 0
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		pages++
		t.Logf("Page %v", page)
		images, err := images.ExtractImages(page)
		if err != nil {
			return false, err
		}

		for _, i := range images {
			t.Logf("%s\t%s\t%s\t%s\t%v\t\n", i.ID, i.Name, i.Owner, i.Checksum, i.SizeBytes)
			count++
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	t.Logf("--------\n%d images listed on %d pages.\n", count, pages)
	th.AssertEquals(t, 3, pages)
	th.AssertEquals(t, 3, count)
}

func TestAllPagesImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageListSuccessfully(t)

	pages, err := images.List(fakeclient.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	images, err := images.ExtractImages(pages)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 3, len(images))
}

func TestCreateImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageCreationSuccessfully(t)

	id := "e7db3b45-8db7-47ad-8109-3fb55c2c24fd"
	name := "Ubuntu 12.10"

	actualImage, err := images.Create(fakeclient.ServiceClient(), images.CreateOpts{
		ID:   id,
		Name: name,
		Properties: map[string]string{
			"architecture": "x86_64",
		},
		Tags: []string{"ubuntu", "quantal"},
	}).Extract()

	th.AssertNoErr(t, err)

	containerFormat := "bare"
	diskFormat := "qcow2"
	owner := "b4eedccc6fb74fa8a7ad6b08382b852b"
	minDiskGigabytes := 0
	minRAMMegabytes := 0
	file := actualImage.File
	createdDate := actualImage.CreatedAt
	lastUpdate := actualImage.UpdatedAt
	schema := "/v2/schemas/image"

	expectedImage := images.Image{
		ID:   "e7db3b45-8db7-47ad-8109-3fb55c2c24fd",
		Name: "Ubuntu 12.10",
		Tags: []string{"ubuntu", "quantal"},

		Status: images.ImageStatusQueued,

		ContainerFormat: containerFormat,
		DiskFormat:      diskFormat,

		MinDiskGigabytes: minDiskGigabytes,
		MinRAMMegabytes:  minRAMMegabytes,

		Owner: owner,

		Visibility:  images.ImageVisibilityPrivate,
		File:        file,
		CreatedAt:   createdDate,
		UpdatedAt:   lastUpdate,
		Schema:      schema,
		VirtualSize: 0,
		Properties: map[string]interface{}{
			"hw_disk_bus":       "scsi",
			"hw_disk_bus_model": "virtio-scsi",
			"hw_scsi_model":     "virtio-scsi",
		},
	}

	th.AssertDeepEquals(t, &expectedImage, actualImage)
}

func TestCreateImageNulls(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageCreationSuccessfullyNulls(t)

	id := "e7db3b45-8db7-47ad-8109-3fb55c2c24fd"
	name := "Ubuntu 12.10"

	actualImage, err := images.Create(fakeclient.ServiceClient(), images.CreateOpts{
		ID:   id,
		Name: name,
		Tags: []string{"ubuntu", "quantal"},
	}).Extract()

	th.AssertNoErr(t, err)

	containerFormat := "bare"
	diskFormat := "qcow2"
	owner := "b4eedccc6fb74fa8a7ad6b08382b852b"
	minDiskGigabytes := 0
	minRAMMegabytes := 0
	file := actualImage.File
	createdDate := actualImage.CreatedAt
	lastUpdate := actualImage.UpdatedAt
	schema := "/v2/schemas/image"

	expectedImage := images.Image{
		ID:   "e7db3b45-8db7-47ad-8109-3fb55c2c24fd",
		Name: "Ubuntu 12.10",
		Tags: []string{"ubuntu", "quantal"},

		Status: images.ImageStatusQueued,

		ContainerFormat: containerFormat,
		DiskFormat:      diskFormat,

		MinDiskGigabytes: minDiskGigabytes,
		MinRAMMegabytes:  minRAMMegabytes,

		Owner: owner,

		Visibility: images.ImageVisibilityPrivate,
		File:       file,
		CreatedAt:  createdDate,
		UpdatedAt:  lastUpdate,
		Schema:     schema,
	}

	th.AssertDeepEquals(t, &expectedImage, actualImage)
}

func TestGetImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageGetSuccessfully(t)

	actualImage, err := images.Get(fakeclient.ServiceClient(), "1bea47ed-f6a9-463b-b423-14b9cca9ad27").Extract()

	th.AssertNoErr(t, err)

	checksum := "64d7c1cd2b6f60c92c14662941cb7913"
	sizeBytes := int64(13167616)
	containerFormat := "bare"
	diskFormat := "qcow2"
	minDiskGigabytes := 0
	minRAMMegabytes := 0
	owner := "5ef70662f8b34079a6eddb8da9d75fe8"
	file := actualImage.File
	createdDate := actualImage.CreatedAt
	lastUpdate := actualImage.UpdatedAt
	schema := "/v2/schemas/image"

	expectedImage := images.Image{
		ID:   "1bea47ed-f6a9-463b-b423-14b9cca9ad27",
		Name: "cirros-0.3.2-x86_64-disk",
		Tags: []string{},

		Status: images.ImageStatusActive,

		ContainerFormat: containerFormat,
		DiskFormat:      diskFormat,

		MinDiskGigabytes: minDiskGigabytes,
		MinRAMMegabytes:  minRAMMegabytes,

		Owner: owner,

		Protected:  false,
		Visibility: images.ImageVisibilityPublic,

		Checksum:    checksum,
		SizeBytes:   sizeBytes,
		File:        file,
		CreatedAt:   createdDate,
		UpdatedAt:   lastUpdate,
		Schema:      schema,
		VirtualSize: 0,
		Properties: map[string]interface{}{
			"hw_disk_bus":       "scsi",
			"hw_disk_bus_model": "virtio-scsi",
			"hw_scsi_model":     "virtio-scsi",
		},
	}

	th.AssertDeepEquals(t, &expectedImage, actualImage)
}

func TestDeleteImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageDeleteSuccessfully(t)

	result := images.Delete(fakeclient.ServiceClient(), "1bea47ed-f6a9-463b-b423-14b9cca9ad27")
	th.AssertNoErr(t, result.Err)
}

func TestUpdateImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageUpdateSuccessfully(t)

	actualImage, err := images.Update(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea", images.UpdateOpts{
		images.ReplaceImageName{NewName: "Fedora 17"},
		images.ReplaceImageTags{NewTags: []string{"fedora", "beefy"}},
	}).Extract()

	th.AssertNoErr(t, err)

	sizebytes := int64(2254249)
	checksum := "2cec138d7dae2aa59038ef8c9aec2390"
	file := actualImage.File
	createdDate := actualImage.CreatedAt
	lastUpdate := actualImage.UpdatedAt
	schema := "/v2/schemas/image"

	expectedImage := images.Image{
		ID:         "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		Name:       "Fedora 17",
		Status:     images.ImageStatusActive,
		Visibility: images.ImageVisibilityPublic,

		SizeBytes: sizebytes,
		Checksum:  checksum,

		Tags: []string{
			"fedora",
			"beefy",
		},

		Owner:            "",
		MinRAMMegabytes:  0,
		MinDiskGigabytes: 0,

		DiskFormat:      "",
		ContainerFormat: "",
		File:            file,
		CreatedAt:       createdDate,
		UpdatedAt:       lastUpdate,
		Schema:          schema,
		VirtualSize:     0,
		Properties: map[string]interface{}{
			"hw_disk_bus":       "scsi",
			"hw_disk_bus_model": "virtio-scsi",
			"hw_scsi_model":     "virtio-scsi",
		},
	}

	th.AssertDeepEquals(t, &expectedImage, actualImage)
}
