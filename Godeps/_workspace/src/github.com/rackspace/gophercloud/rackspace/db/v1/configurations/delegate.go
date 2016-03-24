package configurations

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/db/v1/configurations"
	"github.com/rackspace/gophercloud/pagination"
)

// List will list all of the available configurations.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client)
}

// Create will create a new configuration group.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(client, opts)
}

// Get will retrieve the details for a specified configuration group.
func Get(client *gophercloud.ServiceClient, configID string) os.GetResult {
	return os.Get(client, configID)
}

// Update will modify an existing configuration group by performing a merge
// between new and existing values. If the key already exists, the new value
// will overwrite. All other keys will remain unaffected.
func Update(client *gophercloud.ServiceClient, configID string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(client, configID, opts)
}

// Replace will modify an existing configuration group by overwriting the
// entire parameter group with the new values provided. Any existing keys not
// included in UpdateOptsBuilder will be deleted.
func Replace(client *gophercloud.ServiceClient, configID string, opts os.UpdateOptsBuilder) os.ReplaceResult {
	return os.Replace(client, configID, opts)
}

// Delete will permanently delete a configuration group. Please note that
// config groups cannot be deleted whilst still attached to running instances -
// you must detach and then delete them.
func Delete(client *gophercloud.ServiceClient, configID string) os.DeleteResult {
	return os.Delete(client, configID)
}

// ListInstances will list all the instances associated with a particular
// configuration group.
func ListInstances(client *gophercloud.ServiceClient, configID string) pagination.Pager {
	return os.ListInstances(client, configID)
}

// ListDatastoreParams will list all the available and supported parameters
// that can be used for a particular datastore ID and a particular version.
// For example, if you are wondering how you can configure a MySQL 5.6 instance,
// you can use this operation (you will need to retrieve the MySQL datastore ID
// by using the datastores API).
func ListDatastoreParams(client *gophercloud.ServiceClient, datastoreID, versionID string) pagination.Pager {
	return os.ListDatastoreParams(client, datastoreID, versionID)
}

// GetDatastoreParam will retrieve information about a specific configuration
// parameter. For example, you can use this operation to understand more about
// "innodb_file_per_table" configuration param for MySQL datastores. You will
// need the param's ID first, which can be attained by using the ListDatastoreParams
// operation.
func GetDatastoreParam(client *gophercloud.ServiceClient, datastoreID, versionID, paramID string) os.ParamResult {
	return os.GetDatastoreParam(client, datastoreID, versionID, paramID)
}

// ListGlobalParams is similar to ListDatastoreParams but does not require a
// DatastoreID.
func ListGlobalParams(client *gophercloud.ServiceClient, versionID string) pagination.Pager {
	return os.ListGlobalParams(client, versionID)
}

// GetGlobalParam is similar to GetDatastoreParam but does not require a
// DatastoreID.
func GetGlobalParam(client *gophercloud.ServiceClient, versionID, paramID string) os.ParamResult {
	return os.GetGlobalParam(client, versionID, paramID)
}
