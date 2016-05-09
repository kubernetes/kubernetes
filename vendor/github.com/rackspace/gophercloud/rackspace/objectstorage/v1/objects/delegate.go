package objects

import (
	"io"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/objects"
	"github.com/rackspace/gophercloud/pagination"
)

// ExtractInfo is a function that takes a page of objects and returns their full information.
func ExtractInfo(page pagination.Page) ([]os.Object, error) {
	return os.ExtractInfo(page)
}

// ExtractNames is a function that takes a page of objects and returns only their names.
func ExtractNames(page pagination.Page) ([]string, error) {
	return os.ExtractNames(page)
}

// List is a function that retrieves objects in the container as
// well as container metadata. It returns a pager which can be iterated with the
// EachPage function.
func List(c *gophercloud.ServiceClient, containerName string, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, containerName, opts)
}

// Download is a function that retrieves the content and metadata for an object.
// To extract just the content, pass the DownloadResult response to the
// ExtractContent function.
func Download(c *gophercloud.ServiceClient, containerName, objectName string, opts os.DownloadOptsBuilder) os.DownloadResult {
	return os.Download(c, containerName, objectName, opts)
}

// Create is a function that creates a new object or replaces an existing object.
func Create(c *gophercloud.ServiceClient, containerName, objectName string, content io.ReadSeeker, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, containerName, objectName, content, opts)
}

// CopyOpts is a structure that holds parameters for copying one object to
// another.
type CopyOpts struct {
	Metadata           map[string]string
	ContentDisposition string `h:"Content-Disposition"`
	ContentEncoding    string `h:"Content-Encoding"`
	ContentLength      int    `h:"Content-Length"`
	ContentType        string `h:"Content-Type"`
	CopyFrom           string `h:"X-Copy_From"`
	Destination        string `h:"Destination"`
	DetectContentType  bool   `h:"X-Detect-Content-Type"`
}

// ToObjectCopyMap formats a CopyOpts into a map of headers.
func (opts CopyOpts) ToObjectCopyMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Object-Meta-"+k] = v
	}
	// `Content-Length` is required and a value of "0" is acceptable, but calling `gophercloud.BuildHeaders`
	// will remove the `Content-Length` header if it's set to 0 (or equivalently not set). This will add
	// the header if it's not already set.
	if _, ok := h["Content-Length"]; !ok {
		h["Content-Length"] = "0"
	}
	return h, nil
}

// Copy is a function that copies one object to another.
func Copy(c *gophercloud.ServiceClient, containerName, objectName string, opts os.CopyOptsBuilder) os.CopyResult {
	return os.Copy(c, containerName, objectName, opts)
}

// Delete is a function that deletes an object.
func Delete(c *gophercloud.ServiceClient, containerName, objectName string, opts os.DeleteOptsBuilder) os.DeleteResult {
	return os.Delete(c, containerName, objectName, opts)
}

// Get is a function that retrieves the metadata of an object. To extract just the custom
// metadata, pass the GetResult response to the ExtractMetadata function.
func Get(c *gophercloud.ServiceClient, containerName, objectName string, opts os.GetOptsBuilder) os.GetResult {
	return os.Get(c, containerName, objectName, opts)
}

// Update is a function that creates, updates, or deletes an object's metadata.
func Update(c *gophercloud.ServiceClient, containerName, objectName string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, containerName, objectName, opts)
}

func CreateTempURL(c *gophercloud.ServiceClient, containerName, objectName string, opts os.CreateTempURLOpts) (string, error) {
	return os.CreateTempURL(c, containerName, objectName, opts)
}
