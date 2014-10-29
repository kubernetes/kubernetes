package cdncontainers

import (
	"strconv"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/pagination"
)

// ExtractNames interprets a page of List results when just the container
// names are requested.
func ExtractNames(page pagination.Page) ([]string, error) {
	return os.ExtractNames(page)
}

// ListOpts are options for listing Rackspace CDN containers.
type ListOpts struct {
	EndMarker string `q:"end_marker"`
	Format    string `q:"format"`
	Limit     int    `q:"limit"`
	Marker    string `q:"marker"`
}

// ToContainerListParams formats a ListOpts into a query string and boolean
// representing whether to list complete information for each container.
func (opts ListOpts) ToContainerListParams() (bool, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return false, "", err
	}
	return false, q.String(), nil
}

// List is a function that retrieves containers associated with the account as
// well as account metadata. It returns a pager which can be iterated with the
// EachPage function.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// Get is a function that retrieves the metadata of a container. To extract just
// the custom metadata, pass the GetResult response to the ExtractMetadata
// function.
func Get(c *gophercloud.ServiceClient, containerName string) os.GetResult {
	return os.Get(c, containerName)
}

// UpdateOpts is a structure that holds parameters for updating, creating, or
// deleting a container's metadata.
type UpdateOpts struct {
	CDNEnabled   bool `h:"X-Cdn-Enabled"`
	LogRetention bool `h:"X-Log-Retention"`
	TTL          int  `h:"X-Ttl"`
}

// ToContainerUpdateMap formats a CreateOpts into a map of headers.
func (opts UpdateOpts) ToContainerUpdateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	h["X-Cdn-Enabled"] = strconv.FormatBool(opts.CDNEnabled)
	return h, nil
}

// Update is a function that creates, updates, or deletes a container's
// metadata.
func Update(c *gophercloud.ServiceClient, containerName string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, containerName, opts)
}
