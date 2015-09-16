package objects

import (
	"strings"
	"testing"

	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/objects"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestDownloadObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDownloadObjectSuccessfully(t)

	content, err := Download(fake.ServiceClient(), "testContainer", "testObject", nil).ExtractContent()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, string(content), "Successful download with Gophercloud")
}

func TestListObjectsInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListObjectsInfoSuccessfully(t)

	count := 0
	options := &os.ListOpts{Full: true}
	err := List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractInfo(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, os.ExpectedListInfo, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListObjectNames(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListObjectNamesSuccessfully(t)

	count := 0
	options := &os.ListOpts{Full: false}
	err := List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNames(page)
		if err != nil {
			t.Errorf("Failed to extract container names: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, os.ExpectedListNames, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestCreateObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	
        content := "Did gyre and gimble in the wabe"
        os.HandleCreateTextObjectSuccessfully(t, content)

	options := &os.CreateOpts{ContentType: "text/plain"}
	res := Create(fake.ServiceClient(), "testContainer", "testObject", strings.NewReader(content), options)
	th.AssertNoErr(t, res.Err)
}

func TestCreateObjectWithoutContentType(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

        content := "The sky was the color of television, tuned to a dead channel."
	os.HandleCreateTypelessObjectSuccessfully(t, content)

	res := Create(fake.ServiceClient(), "testContainer", "testObject", strings.NewReader(content), &os.CreateOpts{})
	th.AssertNoErr(t, res.Err)
}

func TestCopyObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCopyObjectSuccessfully(t)

	options := &CopyOpts{Destination: "/newTestContainer/newTestObject"}
	res := Copy(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestDeleteObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDeleteObjectSuccessfully(t)

	res := Delete(fake.ServiceClient(), "testContainer", "testObject", nil)
	th.AssertNoErr(t, res.Err)
}

func TestUpdateObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleUpdateObjectSuccessfully(t)

	options := &os.UpdateOpts{Metadata: map[string]string{"Gophercloud-Test": "objects"}}
	res := Update(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestGetObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetObjectSuccessfully(t)

	expected := map[string]string{"Gophercloud-Test": "objects"}
	actual, err := Get(fake.ServiceClient(), "testContainer", "testObject", nil).ExtractMetadata()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}
