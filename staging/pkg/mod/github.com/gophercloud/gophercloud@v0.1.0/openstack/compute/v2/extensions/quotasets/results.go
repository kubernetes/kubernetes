package quotasets

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// QuotaSet is a set of operational limits that allow for control of compute
// usage.
type QuotaSet struct {
	// ID is tenant associated with this QuotaSet.
	ID string `json:"id"`

	// FixedIPs is number of fixed ips alloted this QuotaSet.
	FixedIPs int `json:"fixed_ips"`

	// FloatingIPs is number of floating ips alloted this QuotaSet.
	FloatingIPs int `json:"floating_ips"`

	// InjectedFileContentBytes is the allowed bytes for each injected file.
	InjectedFileContentBytes int `json:"injected_file_content_bytes"`

	// InjectedFilePathBytes is allowed bytes for each injected file path.
	InjectedFilePathBytes int `json:"injected_file_path_bytes"`

	// InjectedFiles is the number of injected files allowed for each project.
	InjectedFiles int `json:"injected_files"`

	// KeyPairs is number of ssh keypairs.
	KeyPairs int `json:"key_pairs"`

	// MetadataItems is number of metadata items allowed for each instance.
	MetadataItems int `json:"metadata_items"`

	// RAM is megabytes allowed for each instance.
	RAM int `json:"ram"`

	// SecurityGroupRules is number of security group rules allowed for each
	// security group.
	SecurityGroupRules int `json:"security_group_rules"`

	// SecurityGroups is the number of security groups allowed for each project.
	SecurityGroups int `json:"security_groups"`

	// Cores is number of instance cores allowed for each project.
	Cores int `json:"cores"`

	// Instances is number of instances allowed for each project.
	Instances int `json:"instances"`

	// ServerGroups is the number of ServerGroups allowed for the project.
	ServerGroups int `json:"server_groups"`

	// ServerGroupMembers is the number of members for each ServerGroup.
	ServerGroupMembers int `json:"server_group_members"`
}

// QuotaDetailSet represents details of both operational limits of compute
// resources and the current usage of those resources.
type QuotaDetailSet struct {
	// ID is the tenant ID associated with this QuotaDetailSet.
	ID string `json:"id"`

	// FixedIPs is number of fixed ips alloted this QuotaDetailSet.
	FixedIPs QuotaDetail `json:"fixed_ips"`

	// FloatingIPs is number of floating ips alloted this QuotaDetailSet.
	FloatingIPs QuotaDetail `json:"floating_ips"`

	// InjectedFileContentBytes is the allowed bytes for each injected file.
	InjectedFileContentBytes QuotaDetail `json:"injected_file_content_bytes"`

	// InjectedFilePathBytes is allowed bytes for each injected file path.
	InjectedFilePathBytes QuotaDetail `json:"injected_file_path_bytes"`

	// InjectedFiles is the number of injected files allowed for each project.
	InjectedFiles QuotaDetail `json:"injected_files"`

	// KeyPairs is number of ssh keypairs.
	KeyPairs QuotaDetail `json:"key_pairs"`

	// MetadataItems is number of metadata items allowed for each instance.
	MetadataItems QuotaDetail `json:"metadata_items"`

	// RAM is megabytes allowed for each instance.
	RAM QuotaDetail `json:"ram"`

	// SecurityGroupRules is number of security group rules allowed for each
	// security group.
	SecurityGroupRules QuotaDetail `json:"security_group_rules"`

	// SecurityGroups is the number of security groups allowed for each project.
	SecurityGroups QuotaDetail `json:"security_groups"`

	// Cores is number of instance cores allowed for each project.
	Cores QuotaDetail `json:"cores"`

	// Instances is number of instances allowed for each project.
	Instances QuotaDetail `json:"instances"`

	// ServerGroups is the number of ServerGroups allowed for the project.
	ServerGroups QuotaDetail `json:"server_groups"`

	// ServerGroupMembers is the number of members for each ServerGroup.
	ServerGroupMembers QuotaDetail `json:"server_group_members"`
}

// QuotaDetail is a set of details about a single operational limit that allows
// for control of compute usage.
type QuotaDetail struct {
	// InUse is the current number of provisioned/allocated resources of the
	// given type.
	InUse int `json:"in_use"`

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
func (page QuotaSetPage) IsEmpty() (bool, error) {
	ks, err := ExtractQuotaSets(page)
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

// DeleteResult is the response from a Delete operation. Call its Extract method
// to interpret it as a QuotaSet.
type DeleteResult struct {
	quotaResult
}

type quotaDetailResult struct {
	gophercloud.Result
}

// GetDetailResult is the response from a Get operation. Call its Extract
// method to interpret it as a QuotaSet.
type GetDetailResult struct {
	quotaDetailResult
}

// Extract is a method that attempts to interpret any QuotaDetailSet
// resource response as a set of QuotaDetailSet structs.
func (r quotaDetailResult) Extract() (QuotaDetailSet, error) {
	var s struct {
		QuotaData QuotaDetailSet `json:"quota_set"`
	}
	err := r.ExtractInto(&s)
	return s.QuotaData, err
}
