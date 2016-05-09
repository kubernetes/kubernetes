package instances

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	"github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
)

// CreateOptsBuilder is the top-level interface for create options.
type CreateOptsBuilder interface {
	ToInstanceCreateMap() (map[string]interface{}, error)
}

// DatastoreOpts represents the configuration for how an instance stores data.
type DatastoreOpts struct {
	Version string
	Type    string
}

func (opts DatastoreOpts) ToMap() (map[string]string, error) {
	return map[string]string{
		"version": opts.Version,
		"type":    opts.Type,
	}, nil
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
		return nil, fmt.Errorf("Size (GB) must be between 1-300")
	}
	if opts.FlavorRef == "" {
		return nil, fmt.Errorf("FlavorRef is a required field")
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
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToInstanceCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Request("POST", baseURL(client), gophercloud.RequestOpts{
		JSONBody:     &reqBody,
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// List retrieves the status and information for all database instances.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPageFn := func(r pagination.PageResult) pagination.Page {
		return InstancePage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, baseURL(client), createPageFn)
}

// Get retrieves the status and information for a specified database instance.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult

	_, res.Err = client.Request("GET", resourceURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// Delete permanently destroys the database instance.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult

	_, res.Err = client.Request("DELETE", resourceURL(client, id), gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return res
}

// EnableRootUser enables the login from any host for the root user and
// provides the user with a generated root password.
func EnableRootUser(client *gophercloud.ServiceClient, id string) UserRootResult {
	var res UserRootResult

	_, res.Err = client.Request("POST", userRootURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// IsRootEnabled checks an instance to see if root access is enabled. It returns
// True if root user is enabled for the specified database instance or False
// otherwise.
func IsRootEnabled(client *gophercloud.ServiceClient, id string) (bool, error) {
	var res gophercloud.Result

	_, err := client.Request("GET", userRootURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res.Body.(map[string]interface{})["rootEnabled"] == true, err
}

// Restart will restart only the MySQL Instance. Restarting MySQL will
// erase any dynamic configuration settings that you have made within MySQL.
// The MySQL service will be unavailable until the instance restarts.
func Restart(client *gophercloud.ServiceClient, id string) ActionResult {
	var res ActionResult

	_, res.Err = client.Request("POST", actionURL(client, id), gophercloud.RequestOpts{
		JSONBody: map[string]interface{}{"restart": struct{}{}},
		OkCodes:  []int{202},
	})

	return res
}

// Resize changes the memory size of the instance, assuming a valid
// flavorRef is provided. It will also restart the MySQL service.
func Resize(client *gophercloud.ServiceClient, id, flavorRef string) ActionResult {
	var res ActionResult

	type resize struct {
		FlavorRef string `json:"flavorRef"`
	}

	type req struct {
		Resize resize `json:"resize"`
	}

	reqBody := req{Resize: resize{FlavorRef: flavorRef}}

	_, res.Err = client.Request("POST", actionURL(client, id), gophercloud.RequestOpts{
		JSONBody: reqBody,
		OkCodes:  []int{202},
	})

	return res
}

// ResizeVolume will resize the attached volume for an instance. It supports
// only increasing the volume size and does not support decreasing the size.
// The volume size is in gigabytes (GB) and must be an integer.
func ResizeVolume(client *gophercloud.ServiceClient, id string, size int) ActionResult {
	var res ActionResult

	type volume struct {
		Size int `json:"size"`
	}

	type resize struct {
		Volume volume `json:"volume"`
	}

	type req struct {
		Resize resize `json:"resize"`
	}

	reqBody := req{Resize: resize{Volume: volume{Size: size}}}

	_, res.Err = client.Request("POST", actionURL(client, id), gophercloud.RequestOpts{
		JSONBody: reqBody,
		OkCodes:  []int{202},
	})

	return res
}
