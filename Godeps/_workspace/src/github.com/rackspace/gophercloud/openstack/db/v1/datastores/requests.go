package datastores

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List will list all available datastore types that instances can use.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	pageFn := func(r pagination.PageResult) pagination.Page {
		return DatastorePage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, baseURL(client), pageFn)
}

// Get will retrieve the details of a specified datastore type.
func Get(client *gophercloud.ServiceClient, datastoreID string) GetResult {
	var res GetResult

	_, res.Err = client.Request("GET", resourceURL(client, datastoreID), gophercloud.RequestOpts{
		OkCodes:      []int{200},
		JSONResponse: &res.Body,
	})

	return res
}

// ListVersions will list all of the available versions for a specified
// datastore type.
func ListVersions(client *gophercloud.ServiceClient, datastoreID string) pagination.Pager {
	pageFn := func(r pagination.PageResult) pagination.Page {
		return VersionPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, versionsURL(client, datastoreID), pageFn)
}

// GetVersion will retrieve the details of a specified datastore version.
func GetVersion(client *gophercloud.ServiceClient, datastoreID, versionID string) GetVersionResult {
	var res GetVersionResult

	_, res.Err = client.Request("GET", versionURL(client, datastoreID, versionID), gophercloud.RequestOpts{
		OkCodes:      []int{200},
		JSONResponse: &res.Body,
	})

	return res
}
