package quotasets

import (
	"github.com/gophercloud/gophercloud"
)

// Get returns public data about a previously created QuotaSet.
func Get(client *gophercloud.ServiceClient, tenantID string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, tenantID), &res.Body, nil)
	return res
}
