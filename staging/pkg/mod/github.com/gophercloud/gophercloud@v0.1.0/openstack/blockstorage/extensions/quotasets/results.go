package quotasets

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// QuotaSet is a set of operational limits that allow for control of block
// storage usage.
type QuotaSet struct {
	// ID is project associated with this QuotaSet.
	ID string `json:"id"`

	// Volumes is the number of volumes that are allowed for each project.
	Volumes int `json:"volumes"`

	// Snapshots is the number of snapshots that are allowed for each project.
	Snapshots int `json:"snapshots"`

	// Gigabytes is the size (GB) of volumes and snapshots that are allowed for
	// each project.
	Gigabytes int `json:"gigabytes"`

	// PerVolumeGigabytes is the size (GB) of volumes and snapshots that are
	// allowed for each project and the specifed volume type.
	PerVolumeGigabytes int `json:"per_volume_gigabytes"`

	// Backups is the number of backups that are allowed for each project.
	Backups int `json:"backups"`

	// BackupGigabytes is the size (GB) of backups that are allowed for each
	// project.
	BackupGigabytes int `json:"backup_gigabytes"`
}

// QuotaUsageSet represents details of both operational limits of block
// storage resources and the current usage of those resources.
type QuotaUsageSet struct {
	// ID is the project ID associated with this QuotaUsageSet.
	ID string `json:"id"`

	// Volumes is the volume usage information for this project, including
	// in_use, limit, reserved and allocated attributes. Note: allocated
	// attribute is available only when nested quota is enabled.
	Volumes QuotaUsage `json:"volumes"`

	// Snapshots is the snapshot usage information for this project, including
	// in_use, limit, reserved and allocated attributes. Note: allocated
	// attribute is available only when nested quota is enabled.
	Snapshots QuotaUsage `json:"snapshots"`

	// Gigabytes is the size (GB) usage information of volumes and snapshots
	// for this project, including in_use, limit, reserved and allocated
	// attributes. Note: allocated attribute is available only when nested
	// quota is enabled.
	Gigabytes QuotaUsage `json:"gigabytes"`

	// PerVolumeGigabytes is the size (GB) usage information for each volume,
	// including in_use, limit, reserved and allocated attributes. Note:
	// allocated attribute is available only when nested quota is enabled and
	// only limit is meaningful here.
	PerVolumeGigabytes QuotaUsage `json:"per_volume_gigabytes"`

	// Backups is the backup usage information for this project, including
	// in_use, limit, reserved and allocated attributes. Note: allocated
	// attribute is available only when nested quota is enabled.
	Backups QuotaUsage `json:"backups"`

	// BackupGigabytes is the size (GB) usage information of backup for this
	// project, including in_use, limit, reserved and allocated attributes.
	// Note: allocated attribute is available only when nested quota is
	// enabled.
	BackupGigabytes QuotaUsage `json:"backup_gigabytes"`
}

// QuotaUsage is a set of details about a single operational limit that allows
// for control of block storage usage.
type QuotaUsage struct {
	// InUse is the current number of provisioned resources of the given type.
	InUse int `json:"in_use"`

	// Allocated is the current number of resources of a given type allocated
	// for use.  It is only available when nested quota is enabled.
	Allocated int `json:"allocated"`

	// Reserved is a transitional state when a claim against quota has been made
	// but the resource is not yet fully online.
	Reserved int `json:"reserved"`

	// Limit is the maximum number of a given resource that can be
	// allocated/provisioned.  This is what "quota" usually refers to.
	Limit int `json:"limit"`
}

// QuotaSetPage stores a single page of all QuotaSet results from a List call.
type QuotaSetPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a QuotaSetsetPage is empty.
func (r QuotaSetPage) IsEmpty() (bool, error) {
	ks, err := ExtractQuotaSets(r)
	return len(ks) == 0, err
}

// ExtractQuotaSets interprets a page of results as a slice of QuotaSets.
func ExtractQuotaSets(r pagination.Page) ([]QuotaSet, error) {
	var s struct {
		QuotaSets []QuotaSet `json:"quotas"`
	}
	err := (r.(QuotaSetPage)).ExtractInto(&s)
	return s.QuotaSets, err
}

type quotaResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any QuotaSet resource response
// as a QuotaSet struct.
func (r quotaResult) Extract() (*QuotaSet, error) {
	var s struct {
		QuotaSet *QuotaSet `json:"quota_set"`
	}
	err := r.ExtractInto(&s)
	return s.QuotaSet, err
}

// GetResult is the response from a Get operation. Call its Extract method to
// interpret it as a QuotaSet.
type GetResult struct {
	quotaResult
}

// UpdateResult is the response from a Update operation. Call its Extract method
// to interpret it as a QuotaSet.
type UpdateResult struct {
	quotaResult
}

type quotaUsageResult struct {
	gophercloud.Result
}

// GetUsageResult is the response from a Get operation. Call its Extract
// method to interpret it as a QuotaSet.
type GetUsageResult struct {
	quotaUsageResult
}

// Extract is a method that attempts to interpret any QuotaUsageSet resource
// response as a set of QuotaUsageSet structs.
func (r quotaUsageResult) Extract() (QuotaUsageSet, error) {
	var s struct {
		QuotaUsageSet QuotaUsageSet `json:"quota_set"`
	}
	err := r.ExtractInto(&s)
	return s.QuotaUsageSet, err
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr
// method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
