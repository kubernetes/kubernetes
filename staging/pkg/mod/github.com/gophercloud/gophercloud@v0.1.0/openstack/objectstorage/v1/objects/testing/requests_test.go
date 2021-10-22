package testing

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/objects"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

var (
	loc, _ = time.LoadLocation("GMT")
)

func TestDownloadReader(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDownloadObjectSuccessfully(t)

	response := objects.Download(fake.ServiceClient(), "testContainer", "testObject", nil)
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

	response := objects.Download(fake.ServiceClient(), "testContainer", "testObject", nil)

	// Check []byte extraction
	bytes, err := response.ExtractContent()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "Successful download with Gophercloud", string(bytes))

	expected := &objects.DownloadHeader{
		ContentLength:     36,
		ContentType:       "text/plain; charset=utf-8",
		Date:              time.Date(2009, time.November, 10, 23, 0, 0, 0, loc),
		StaticLargeObject: true,
	}
	actual, err := response.Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}

func TestListObjectInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListObjectsInfoSuccessfully(t)

	count := 0
	options := &objects.ListOpts{Full: true}
	err := objects.List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := objects.ExtractInfo(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedListInfo, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListObjectSubdir(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSubdirSuccessfully(t)

	count := 0
	options := &objects.ListOpts{Full: true, Prefix: "", Delimiter: "/"}
	err := objects.List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := objects.ExtractInfo(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedListSubdir, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListObjectNames(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListObjectNamesSuccessfully(t)

	// Check without delimiter.
	count := 0
	options := &objects.ListOpts{Full: false}
	err := objects.List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := objects.ExtractNames(page)
		if err != nil {
			t.Errorf("Failed to extract container names: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, ExpectedListNames, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)

	// Check with delimiter.
	count = 0
	options = &objects.ListOpts{Full: false, Delimiter: "/"}
	err = objects.List(fake.ServiceClient(), "testContainer", options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := objects.ExtractNames(page)
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

	content := "Did gyre and gimble in the wabe"

	HandleCreateTextObjectSuccessfully(t, content)

	options := &objects.CreateOpts{ContentType: "text/plain", Content: strings.NewReader(content)}
	res := objects.Create(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestCreateObjectWithCacheControl(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	content := "All mimsy were the borogoves"

	HandleCreateTextWithCacheControlSuccessfully(t, content)

	options := &objects.CreateOpts{
		CacheControl: `max-age="3600", public`,
		Content:      strings.NewReader(content),
	}
	res := objects.Create(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestCreateObjectWithoutContentType(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	content := "The sky was the color of television, tuned to a dead channel."

	HandleCreateTypelessObjectSuccessfully(t, content)

	res := objects.Create(fake.ServiceClient(), "testContainer", "testObject", &objects.CreateOpts{Content: strings.NewReader(content)})
	th.AssertNoErr(t, res.Err)
}

/*
func TestErrorIsRaisedForChecksumMismatch(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", "acbd18db4cc2f85cedef654fccc4a4d8")
		w.WriteHeader(http.StatusCreated)
	})

	content := strings.NewReader("The sky was the color of television, tuned to a dead channel.")
	res := Create(fake.ServiceClient(), "testContainer", "testObject", &CreateOpts{Content: content})

	err := fmt.Errorf("Local checksum does not match API ETag header")
	th.AssertDeepEquals(t, err, res.Err)
}
*/

func TestCopyObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCopyObjectSuccessfully(t)

	options := &objects.CopyOpts{Destination: "/newTestContainer/newTestObject"}
	res := objects.Copy(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestDeleteObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteObjectSuccessfully(t)

	res := objects.Delete(fake.ServiceClient(), "testContainer", "testObject", nil)
	th.AssertNoErr(t, res.Err)
}

func TestUpateObjectMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateObjectSuccessfully(t)

	options := &objects.UpdateOpts{Metadata: map[string]string{"Gophercloud-Test": "objects"}}
	res := objects.Update(fake.ServiceClient(), "testContainer", "testObject", options)
	th.AssertNoErr(t, res.Err)
}

func TestGetObject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetObjectSuccessfully(t)

	expected := map[string]string{"Gophercloud-Test": "objects"}
	actual, err := objects.Get(fake.ServiceClient(), "testContainer", "testObject", nil).ExtractMetadata()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)

	getOpts := objects.GetOpts{
		Newest: true,
	}
	actualHeaders, err := objects.Get(fake.ServiceClient(), "testContainer", "testObject", getOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, actualHeaders.StaticLargeObject, true)
}

func TestETag(t *testing.T) {
	content := "some example object"
	createOpts := objects.CreateOpts{
		Content: strings.NewReader(content),
		NoETag:  true,
	}

	_, headers, _, err := createOpts.ToObjectCreateParams()
	th.AssertNoErr(t, err)
	_, ok := headers["ETag"]
	th.AssertEquals(t, ok, false)

	hash := md5.New()
	io.WriteString(hash, content)
	localChecksum := fmt.Sprintf("%x", hash.Sum(nil))

	createOpts = objects.CreateOpts{
		Content: strings.NewReader(content),
		ETag:    localChecksum,
	}

	_, headers, _, err = createOpts.ToObjectCreateParams()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, headers["ETag"], localChecksum)
}

func TestObjectCreateParamsWithoutSeek(t *testing.T) {
	content := "I do not implement Seek()"
	buf := bytes.NewBuffer([]byte(content))

	createOpts := objects.CreateOpts{Content: buf}
	reader, headers, _, err := createOpts.ToObjectCreateParams()

	th.AssertNoErr(t, err)

	_, ok := reader.(io.ReadSeeker)
	th.AssertEquals(t, ok, true)

	c, err := ioutil.ReadAll(reader)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, content, string(c))

	_, ok = headers["ETag"]
	th.AssertEquals(t, true, ok)
}

func TestObjectCreateParamsWithSeek(t *testing.T) {
	content := "I implement Seek()"
	createOpts := objects.CreateOpts{Content: strings.NewReader(content)}
	reader, headers, _, err := createOpts.ToObjectCreateParams()

	th.AssertNoErr(t, err)

	_, ok := reader.(io.ReadSeeker)
	th.AssertEquals(t, ok, true)

	c, err := ioutil.ReadAll(reader)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, content, string(c))

	_, ok = headers["ETag"]
	th.AssertEquals(t, true, ok)
}
