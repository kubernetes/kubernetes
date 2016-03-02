package accounts

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/accounts"
)

// Get is a function that retrieves an account's metadata. To extract just the
// custom metadata, call the ExtractMetadata method on the GetResult. To extract
// all the headers that are returned (including the metadata), call the
// ExtractHeader method on the GetResult.
func Get(c *gophercloud.ServiceClient) os.GetResult {
	return os.Get(c, nil)
}

// UpdateOpts is a structure that contains parameters for updating, creating, or
// deleting an account's metadata.
type UpdateOpts struct {
	Metadata    map[string]string
	TempURLKey  string `h:"X-Account-Meta-Temp-URL-Key"`
	TempURLKey2 string `h:"X-Account-Meta-Temp-URL-Key-2"`
}

// ToAccountUpdateMap formats an UpdateOpts into a map[string]string of headers.
func (opts UpdateOpts) ToAccountUpdateMap() (map[string]string, error) {
	headers, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		headers["X-Account-Meta-"+k] = v
	}
	return headers, err
}

// Update will update an account's metadata with the Metadata in the UpdateOptsBuilder.
func Update(c *gophercloud.ServiceClient, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, opts)
}
