package quotasets

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
)

// Get returns public data about a previously created QuotaSet.
func Get(client *gophercloud.ServiceClient, projectID string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, projectID), &r.Body, nil)
	return
}

// GetDefaults returns public data about the project's default block storage quotas.
func GetDefaults(client *gophercloud.ServiceClient, projectID string) (r GetResult) {
	_, r.Err = client.Get(getDefaultsURL(client, projectID), &r.Body, nil)
	return
}

// GetUsage returns detailed public data about a previously created QuotaSet.
func GetUsage(client *gophercloud.ServiceClient, projectID string) (r GetUsageResult) {
	u := fmt.Sprintf("%s?usage=true", getURL(client, projectID))
	_, r.Err = client.Get(u, &r.Body, nil)
	return
}

// Updates the quotas for the given projectID and returns the new QuotaSet.
func Update(client *gophercloud.ServiceClient, projectID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToBlockStorageQuotaUpdateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(updateURL(client, projectID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return r
}

// UpdateOptsBuilder enables extensins to add parameters to the update request.
type UpdateOptsBuilder interface {
	// Extra specific name to prevent collisions with interfaces for other quotas
	// (e.g. neutron)
	ToBlockStorageQuotaUpdateMap() (map[string]interface{}, error)
}

// ToBlockStorageQuotaUpdateMap builds the update options into a serializable
// format.
func (opts UpdateOpts) ToBlockStorageQuotaUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "quota_set")
}

// Options for Updating the quotas of a Tenant.
// All int-values are pointers so they can be nil if they are not needed.
// You can use gopercloud.IntToPointer() for convenience
type UpdateOpts struct {
	// Volumes is the number of volumes that are allowed for each project.
	Volumes *int `json:"volumes,omitempty"`

	// Snapshots is the number of snapshots that are allowed for each project.
	Snapshots *int `json:"snapshots,omitempty"`

	// Gigabytes is the size (GB) of volumes and snapshots that are allowed for
	// each project.
	Gigabytes *int `json:"gigabytes,omitempty"`

	// PerVolumeGigabytes is the size (GB) of volumes and snapshots that are
	// allowed for each project and the specifed volume type.
	PerVolumeGigabytes *int `json:"per_volume_gigabytes,omitempty"`

	// Backups is the number of backups that are allowed for each project.
	Backups *int `json:"backups,omitempty"`

	// BackupGigabytes is the size (GB) of backups that are allowed for each
	// project.
	BackupGigabytes *int `json:"backup_gigabytes,omitempty"`

	// Groups is the number of groups that are allowed for each project.
	Groups *int `json:"groups,omitempty"`

	// Force will update the quotaset even if the quota has already been used
	// and the reserved quota exceeds the new quota.
	Force bool `json:"force,omitempty"`
}

// Resets the quotas for the given tenant to their default values.
func Delete(client *gophercloud.ServiceClient, projectID string) (r DeleteResult) {
	_, r.Err = client.Delete(updateURL(client, projectID), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
