package objects

import (
	"bytes"
	"io"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestDownloadReader(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDownloadObjectSuccessfully(t)

	response := Download(fake.ServiceClient(), "testContainer", "testObject", nil)
	defer response.Body.Close()

	// Check reader
	buf := bytes.NewBuffer(make([]byte, 0))
	io.CopyN(buf, response.Body, 10)
	th.CheckEquals(t, "Successful", string(buf.Bytes()))
}

func TestDownloadExtraction(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDownloadObjectSuccessfully(t)

	response := Download(fake.ServiceClient(), "testContainer", "testObject", nil)

	// Check []byte extraction
	bytes, err := response.ExtractContent()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "Successful download with Gophercloud", string(bytes))
}

func TestListObjectInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListObjectsInfoSuccessfully(t)

	count := 0
	options := &ListOpts{Full: true}
	err := List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractInfo(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedListInfo, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListObjectNames(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListObjectNamesSuccessfully(t)

	count := 0
	options := &ListOpts{Full: false}
	err := List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNames(page)
		if err != nil {
			t.Errorf("Failed to extract container names: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, ExpectedListNames, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestCreateObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateTextObjectSuccessfully(t)

	content := bytes.NewBufferString("Did gyre and gimble in the wabe")
	options := &CreateOpts{ContentType: "text/plain"}
	res := Create(fake.ServiceClient(), "testContainer", "testObject", content, options)
	th.AssertNoErr(t, res.Err)
}

func TestCreateObjectWithoutContentType(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateTypelessObjectSuccessfully(t)

	content := bytes.NewBufferString("The sky was the color of television, tuned to a dead channel.")
	res := Create(fake.ServiceClient(), "testContainer", "testObject", content, &CreateOpts{})
	th.AssertNoErr(t, res.Err)
}

func TestCopyObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCopyObjectSuccessfully(t)

	options := &CopyOpts{Destination: "/newTestContainer/newTestObject"}
	res := Copy(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestDeleteObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteObjectSuccessfully(t)

	res := Delete(fake.ServiceClient(), "testContainer", "testObject", nil)
	th.AssertNoErr(t, res.Err)
}

func TestUpateObjectMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateObjectSuccessfully(t)

	options := &UpdateOpts{Metadata: map[string]string{"Gophercloud-Test": "objects"}}
	res := Update(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestGetObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetObjectSuccessfully(t)

	expected := map[string]string{"Gophercloud-Test": "objects"}
	actual, err := Get(fake.ServiceClient(), "testContainer", "testObject", nil).ExtractMetadata()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}
