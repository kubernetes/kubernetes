package migrate

import (
	"github.com/gophercloud/gophercloud"
)

// Migrate will initiate a migration of the instance to another host.
func Migrate(client *gophercloud.ServiceClient, id string) (r MigrateResult) {
	_, r.Err = client.Post(actionURL(client, id), map[string]interface{}{"migrate": nil}, nil, nil)
	return
}
