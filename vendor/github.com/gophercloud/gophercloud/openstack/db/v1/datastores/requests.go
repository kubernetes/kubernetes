package datastores

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List will list all available datastore types that instances can use.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, baseURL(client), func(r pagination.PageResult) pagination.Page {
		return DatastorePage{pagination.SinglePageBase(r)}
	})
}

// Get will retrieve the details of a specified datastore type.
func Get(client *gophercloud.ServiceClient, datastoreID string) (r GetResult) {
	_, r.Err = client.Get(resourceURL(client, datastoreID), &r.Body, nil)
	return
}

// ListVersions will list all of the available versions for a specified
// datastore type.
func ListVersions(client *gophercloud.ServiceClient, datastoreID string) pagination.Pager {
	return pagination.NewPager(client, versionsURL(client, datastoreID), func(r pagination.PageResult) pagination.Page {
		return VersionPage{pagination.SinglePageBase(r)}
	})
}

// GetVersion will retrieve the details of a specified datastore version.
func GetVersion(client *gophercloud.ServiceClient, datastoreID, versionID string) (r GetVersionResult) {
	_, r.Err = client.Get(versionURL(client, datastoreID, versionID), &r.Body, nil)
	return
}
