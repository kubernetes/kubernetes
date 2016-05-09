package instances

import (
	"github.com/rackspace/gophercloud"
	osDBs "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/instances"
	osUsers "github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/db/v1/backups"
)

// CreateOpts is the struct responsible for configuring a new database instance.
type CreateOpts struct {
	// Either the integer UUID (in string form) of the flavor, or its URI
	// reference as specified in the response from the List() call. Required.
	FlavorRef string

	// Specifies the volume size in gigabytes (GB). The value must be between 1
	// and 300. Required.
	Size int

	// Name of the instance to create. The length of the name is limited to
	// 255 characters and any characters are permitted. Optional.
	Name string

	// A slice of database information options.
	Databases osDBs.CreateOptsBuilder

	// A slice of user information options.
	Users osUsers.CreateOptsBuilder

	// ID of the configuration group to associate with the instance. Optional.
	ConfigID string

	// Options to configure the type of datastore the instance will use. This is
	// optional, and if excluded will default to MySQL.
	Datastore *os.DatastoreOpts

	// Specifies the backup ID from which to restore the database instance. There
	// are some things to be aware of before using this field.  When you execute
	// the Restore Backup operation, a new database instance is created to store
	// the backup whose ID is specified by the restorePoint attribute. This will
	// mean that:
	// - All users, passwords and access that were on the instance at the time of
	// the backup will be restored along with the databases.
	// - You can create new users or databases if you want, but they cannot be
	// the same as the ones from the instance that was backed up.
	RestorePoint string

	ReplicaOf string
}

func (opts CreateOpts) ToInstanceCreateMap() (map[string]interface{}, error) {
	instance, err := os.CreateOpts{
		FlavorRef: opts.FlavorRef,
		Size:      opts.Size,
		Name:      opts.Name,
		Databases: opts.Databases,
		Users:     opts.Users,
	}.ToInstanceCreateMap()

	if err != nil {
		return nil, err
	}

	instance = instance["instance"].(map[string]interface{})

	if opts.ConfigID != "" {
		instance["configuration"] = opts.ConfigID
	}

	if opts.Datastore != nil {
		ds, err := opts.Datastore.ToMap()
		if err != nil {
			return nil, err
		}
		instance["datastore"] = ds
	}

	if opts.RestorePoint != "" {
		instance["restorePoint"] = map[string]string{"backupRef": opts.RestorePoint}
	}

	if opts.ReplicaOf != "" {
		instance["replica_of"] = opts.ReplicaOf
	}

	return map[string]interface{}{"instance": instance}, nil
}

// Create asynchronously provisions a new database instance. It requires the
// user to specify a flavor and a volume size. The API service then provisions
// the instance with the requested flavor and sets up a volume of the specified
// size, which is the storage for the database instance.
//
// Although this call only allows the creation of 1 instance per request, you
// can create an instance with multiple databases and users. The default
// binding for a MySQL instance is port 3306.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) CreateResult {
	return CreateResult{os.Create(client, opts)}
}

// ListOpts specifies all of the query options to be used when returning a list
// of database instances.
type ListOpts struct {
	// IncludeHA includes or excludes High Availability instances from the result set
	IncludeHA bool `q:"include_ha"`

	// IncludeReplicas includes or excludes Replica instances from the result set
	IncludeReplicas bool `q:"include_replicas"`
}

// ToInstanceListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToInstanceListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List retrieves the status and information for all database instances.
func List(client *gophercloud.ServiceClient, opts *ListOpts) pagination.Pager {
	url := baseURL(client)

	if opts != nil {
		query, err := opts.ToInstanceListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	createPageFn := func(r pagination.PageResult) pagination.Page {
		return os.InstancePage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, url, createPageFn)
}

// GetDefaultConfig lists the default configuration settings from the template
// that was applied to the specified instance. In a sense, this is the vanilla
// configuration setting applied to an instance. Further configuration can be
// applied by associating an instance with a configuration group.
func GetDefaultConfig(client *gophercloud.ServiceClient, id string) ConfigResult {
	var res ConfigResult

	_, res.Err = client.Request("GET", configURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// AssociateWithConfigGroup associates a specified instance to a specified
// configuration group. If any of the parameters within a configuration group
// require a restart, then the instance will transition into a restart.
func AssociateWithConfigGroup(client *gophercloud.ServiceClient, instanceID, configGroupID string) UpdateResult {
	reqBody := map[string]string{
		"configuration": configGroupID,
	}

	var res UpdateResult

	_, res.Err = client.Request("PUT", resourceURL(client, instanceID), gophercloud.RequestOpts{
		JSONBody: map[string]map[string]string{"instance": reqBody},
		OkCodes:  []int{202},
	})

	return res
}

// DetachFromConfigGroup will detach an instance from all config groups.
func DetachFromConfigGroup(client *gophercloud.ServiceClient, instanceID string) UpdateResult {
	return AssociateWithConfigGroup(client, instanceID, "")
}

// ListBackups will list all the backups for a specified database instance.
func ListBackups(client *gophercloud.ServiceClient, instanceID string) pagination.Pager {
	pageFn := func(r pagination.PageResult) pagination.Page {
		return backups.BackupPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, backupsURL(client, instanceID), pageFn)
}

// DetachReplica will detach a specified replica instance from its source
// instance, effectively allowing it to operate independently. Detaching a
// replica will restart the MySQL service on the instance.
func DetachReplica(client *gophercloud.ServiceClient, replicaID string) DetachResult {
	var res DetachResult

	_, res.Err = client.Request("PATCH", resourceURL(client, replicaID), gophercloud.RequestOpts{
		JSONBody: map[string]interface{}{"instance": map[string]string{"replica_of": "", "slave_of": ""}},
		OkCodes:  []int{202},
	})

	return res
}
