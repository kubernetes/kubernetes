package instances

import (
	"github.com/gophercloud/gophercloud"
	db "github.com/gophercloud/gophercloud/openstack/db/v1/databases"
	"github.com/gophercloud/gophercloud/openstack/db/v1/users"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder is the top-level interface for create options.
type CreateOptsBuilder interface {
	ToInstanceCreateMap() (map[string]interface{}, error)
}

// DatastoreOpts represents the configuration for how an instance stores data.
type DatastoreOpts struct {
	Version string `json:"version"`
	Type    string `json:"type"`
}

// ToMap converts a DatastoreOpts to a map[string]string (for a request body)
func (opts DatastoreOpts) ToMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

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
	Databases db.CreateOptsBuilder
	// A slice of user information options.
	Users users.CreateOptsBuilder
	// Options to configure the type of datastore the instance will use. This is
	// optional, and if excluded will default to MySQL.
	Datastore *DatastoreOpts
}

// ToInstanceCreateMap will render a JSON map.
func (opts CreateOpts) ToInstanceCreateMap() (map[string]interface{}, error) {
	if opts.Size > 300 || opts.Size < 1 {
		err := gophercloud.ErrInvalidInput{}
		err.Argument = "instances.CreateOpts.Size"
		err.Value = opts.Size
		err.Info = "Size (GB) must be between 1-300"
		return nil, err
	}

	if opts.FlavorRef == "" {
		return nil, gophercloud.ErrMissingInput{Argument: "instances.CreateOpts.FlavorRef"}
	}

	instance := map[string]interface{}{
		"volume":    map[string]int{"size": opts.Size},
		"flavorRef": opts.FlavorRef,
	}

	if opts.Name != "" {
		instance["name"] = opts.Name
	}
	if opts.Databases != nil {
		dbs, err := opts.Databases.ToDBCreateMap()
		if err != nil {
			return nil, err
		}
		instance["databases"] = dbs["databases"]
	}
	if opts.Users != nil {
		users, err := opts.Users.ToUserCreateMap()
		if err != nil {
			return nil, err
		}
		instance["users"] = users["users"]
	}
	if opts.Datastore != nil {
		datastore, err := opts.Datastore.ToMap()
		if err != nil {
			return nil, err
		}
		instance["datastore"] = datastore
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
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToInstanceCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(baseURL(client), &b, &r.Body, &gophercloud.RequestOpts{OkCodes: []int{200}})
	return
}

// List retrieves the status and information for all database instances.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, baseURL(client), func(r pagination.PageResult) pagination.Page {
		return InstancePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves the status and information for a specified database instance.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(resourceURL(client, id), &r.Body, nil)
	return
}

// Delete permanently destroys the database instance.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(resourceURL(client, id), nil)
	return
}

// EnableRootUser enables the login from any host for the root user and
// provides the user with a generated root password.
func EnableRootUser(client *gophercloud.ServiceClient, id string) (r EnableRootUserResult) {
	_, r.Err = client.Post(userRootURL(client, id), nil, &r.Body, &gophercloud.RequestOpts{OkCodes: []int{200}})
	return
}

// IsRootEnabled checks an instance to see if root access is enabled. It returns
// True if root user is enabled for the specified database instance or False
// otherwise.
func IsRootEnabled(client *gophercloud.ServiceClient, id string) (r IsRootEnabledResult) {
	_, r.Err = client.Get(userRootURL(client, id), &r.Body, nil)
	return
}

// Restart will restart only the MySQL Instance. Restarting MySQL will
// erase any dynamic configuration settings that you have made within MySQL.
// The MySQL service will be unavailable until the instance restarts.
func Restart(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	b := map[string]interface{}{"restart": struct{}{}}
	_, r.Err = client.Post(actionURL(client, id), &b, nil, nil)
	return
}

// Resize changes the memory size of the instance, assuming a valid
// flavorRef is provided. It will also restart the MySQL service.
func Resize(client *gophercloud.ServiceClient, id, flavorRef string) (r ActionResult) {
	b := map[string]interface{}{"resize": map[string]string{"flavorRef": flavorRef}}
	_, r.Err = client.Post(actionURL(client, id), &b, nil, nil)
	return
}

// ResizeVolume will resize the attached volume for an instance. It supports
// only increasing the volume size and does not support decreasing the size.
// The volume size is in gigabytes (GB) and must be an integer.
func ResizeVolume(client *gophercloud.ServiceClient, id string, size int) (r ActionResult) {
	b := map[string]interface{}{"resize": map[string]interface{}{"volume": map[string]int{"size": size}}}
	_, r.Err = client.Post(actionURL(client, id), &b, nil, nil)
	return
}
