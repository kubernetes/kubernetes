package backups

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToBackupCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains options for creating a Backup. This object is passed to
// the backups.Create function. For more information about these parameters,
// see the Backup object.
type CreateOpts struct {
	// VolumeID is the ID of the volume to create the backup from.
	VolumeID string `json:"volume_id" required:"true"`

	// Force will force the creation of a backup regardless of the
	//volume's status.
	Force bool `json:"force,omitempty"`

	// Name is the name of the backup.
	Name string `json:"name,omitempty"`

	// Description is the description of the backup.
	Description string `json:"description,omitempty"`

	// Metadata is metadata for the backup.
	// Requires microversion 3.43 or later.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Container is a container to store the backup.
	Container string `json:"container,omitempty"`

	// Incremental is whether the backup should be incremental or not.
	Incremental bool `json:"incremental,omitempty"`

	// SnapshotID is the ID of a snapshot to backup.
	SnapshotID string `json:"snapshot_id,omitempty"`

	// AvailabilityZone is an availability zone to locate the volume or snapshot.
	// Requires microversion 3.51 or later.
	AvailabilityZone string `json:"availability_zone,omitempty"`
}

// ToBackupCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToBackupCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "backup")
}

// Create will create a new Backup based on the values in CreateOpts. To
// extract the Backup object from the response, call the Extract method on the
// CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToBackupCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// Delete will delete the existing Backup with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get retrieves the Backup with the provided ID. To extract the Backup
// object from the response, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToBackupListQuery() (string, error)
}

type ListOpts struct {
	// AllTenants will retrieve backups of all tenants/projects.
	AllTenants bool `q:"all_tenants"`

	// Name will filter by the specified backup name.
	// This does not work in later microversions.
	Name string `q:"name"`

	// Status will filter by the specified status.
	// This does not work in later microversions.
	Status string `q:"status"`

	// TenantID will filter by a specific tenant/project ID.
	// Setting AllTenants is required to use this.
	TenantID string `q:"project_id"`

	// VolumeID will filter by a specified volume ID.
	// This does not work in later microversions.
	VolumeID string `q:"volume_id"`

	// Comma-separated list of sort keys and optional sort directions in the
	// form of <key>[:<direction>].
	Sort string `q:"sort"`

	// Requests a page size of items.
	Limit int `q:"limit"`

	// Used in conjunction with limit to return a slice of items.
	Offset int `q:"offset"`

	// The ID of the last-seen item.
	Marker string `q:"marker"`
}

// ToBackupListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToBackupListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns Backups optionally limited by the conditions provided in
// ListOpts.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToBackupListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return BackupPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToBackupUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contain options for updating an existing Backup.
type UpdateOpts struct {
	// Name is the name of the backup.
	Name *string `json:"name,omitempty"`

	// Description is the description of the backup.
	Description *string `json:"description,omitempty"`

	// Metadata is metadata for the backup.
	// Requires microversion 3.43 or later.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ToBackupUpdateMap assembles a request body based on the contents of
// an UpdateOpts.
func (opts UpdateOpts) ToBackupUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Update will update the Backup with provided information. To extract
// the updated Backup from the response, call the Extract method on the
// UpdateResult.
// Requires microversion 3.9 or later.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToBackupUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
