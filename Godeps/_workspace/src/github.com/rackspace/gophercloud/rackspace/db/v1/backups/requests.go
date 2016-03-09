package backups

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// CreateOptsBuilder is the top-level interface for creating JSON maps.
type CreateOptsBuilder interface {
	ToBackupCreateMap() (map[string]interface{}, error)
}

// CreateOpts is responsible for configuring newly provisioned backups.
type CreateOpts struct {
	// [REQUIRED] The name of the backup. The only restriction is the name must
	// be less than 64 characters long.
	Name string

	// [REQUIRED] The ID of the instance being backed up.
	InstanceID string

	// [OPTIONAL] A human-readable explanation of the backup.
	Description string
}

// ToBackupCreateMap will create a JSON map for the Create operation.
func (opts CreateOpts) ToBackupCreateMap() (map[string]interface{}, error) {
	if opts.Name == "" {
		return nil, errors.New("Name is a required field")
	}
	if opts.InstanceID == "" {
		return nil, errors.New("InstanceID is a required field")
	}

	backup := map[string]interface{}{
		"name":     opts.Name,
		"instance": opts.InstanceID,
	}

	if opts.Description != "" {
		backup["description"] = opts.Description
	}

	return map[string]interface{}{"backup": backup}, nil
}

// Create asynchronously creates a new backup for a specified database instance.
// During the backup process, write access on MyISAM databases will be
// temporarily disabled; innoDB databases will be unaffected. During this time,
// you will not be able to add or delete databases or users; nor delete, stop
// or reboot the instance itself. Only one backup is permitted at once.
//
// Backups are not deleted when database instances are deleted; you must
// manually delete any backups created using Delete(). Backups are saved to your
// Cloud Files account in a new container called z_CLOUDDB_BACKUPS. It is
// strongly recommended you do not alter this container or its contents; usual
// storage costs apply.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToBackupCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Request("POST", baseURL(client), gophercloud.RequestOpts{
		JSONBody:     &reqBody,
		JSONResponse: &res.Body,
		OkCodes:      []int{202},
	})

	return res
}

// ListOptsBuilder is the top-level interface for creating query strings.
type ListOptsBuilder interface {
	ToBackupListQuery() (string, error)
}

// ListOpts allows you to refine a list search by certain parameters.
type ListOpts struct {
	// The type of datastore by which to filter.
	Datastore string `q:"datastore"`
}

// ToBackupListQuery converts a ListOpts struct into a query string.
func (opts ListOpts) ToBackupListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List will list all the saved backups for all database instances.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := baseURL(client)

	if opts != nil {
		query, err := opts.ToBackupListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	pageFn := func(r pagination.PageResult) pagination.Page {
		return BackupPage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, url, pageFn)
}

// Get will retrieve details for a particular backup based on its unique ID.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult

	_, res.Err = client.Request("GET", resourceURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// Delete will permanently delete a backup.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult

	_, res.Err = client.Request("DELETE", resourceURL(client, id), gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return res
}
