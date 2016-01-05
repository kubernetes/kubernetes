package containers

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/pagination"
)

// ExtractInfo interprets a page of List results when full container info
// is requested.
func ExtractInfo(page pagination.Page) ([]os.Container, error) {
	return os.ExtractInfo(page)
}

// ExtractNames interprets a page of List results when just the container
// names are requested.
func ExtractNames(page pagination.Page) ([]string, error) {
	return os.ExtractNames(page)
}

// List is a function that retrieves containers associated with the account as
// well as account metadata. It returns a pager which can be iterated with the
// EachPage function.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// CreateOpts is a structure that holds parameters for creating a container.
type CreateOpts struct {
	Metadata         map[string]string
	ContainerRead    string `h:"X-Container-Read"`
	ContainerWrite   string `h:"X-Container-Write"`
	VersionsLocation string `h:"X-Versions-Location"`
}

// ToContainerCreateMap formats a CreateOpts into a map of headers.
func (opts CreateOpts) ToContainerCreateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Container-Meta-"+k] = v
	}
	return h, nil
}

// Create is a function that creates a new container.
func Create(c *gophercloud.ServiceClient, containerName string, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, containerName, opts)
}

// Delete is a function that deletes a container.
func Delete(c *gophercloud.ServiceClient, containerName string) os.DeleteResult {
	return os.Delete(c, containerName)
}

// UpdateOpts is a structure that holds parameters for updating or creating a
// container's metadata.
type UpdateOpts struct {
	Metadata               map[string]string
	ContainerRead          string `h:"X-Container-Read"`
	ContainerWrite         string `h:"X-Container-Write"`
	ContentType            string `h:"Content-Type"`
	DetectContentType      bool   `h:"X-Detect-Content-Type"`
	RemoveVersionsLocation string `h:"X-Remove-Versions-Location"`
	VersionsLocation       string `h:"X-Versions-Location"`
}

// ToContainerUpdateMap formats a CreateOpts into a map of headers.
func (opts UpdateOpts) ToContainerUpdateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Container-Meta-"+k] = v
	}
	return h, nil
}

// Update is a function that creates, updates, or deletes a container's
// metadata.
func Update(c *gophercloud.ServiceClient, containerName string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, containerName, opts)
}

// Get is a function that retrieves the metadata of a container. To extract just
// the custom metadata, pass the GetResult response to the ExtractMetadata
// function.
func Get(c *gophercloud.ServiceClient, containerName string) os.GetResult {
	return os.Get(c, containerName)
}
