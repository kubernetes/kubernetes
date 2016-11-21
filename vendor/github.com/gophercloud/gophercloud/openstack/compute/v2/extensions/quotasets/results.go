package quotasets

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// QuotaSet is a set of operational limits that allow for control of compute usage.
type QuotaSet struct {
	//ID is tenant associated with this quota_set
	ID string `json:"id"`
	//FixedIps is number of fixed ips alloted this quota_set
	FixedIps int `json:"fixed_ips"`
	// FloatingIps is number of floating ips alloted this quota_set
	FloatingIps int `json:"floating_ips"`
	// InjectedFileContentBytes is content bytes allowed for each injected file
	InjectedFileContentBytes int `json:"injected_file_content_bytes"`
	// InjectedFilePathBytes is allowed bytes for each injected file path
	InjectedFilePathBytes int `json:"injected_file_path_bytes"`
	// InjectedFiles is injected files allowed for each project
	InjectedFiles int `json:"injected_files"`
	// KeyPairs is number of ssh keypairs
	KeyPairs int `json:"keypairs"`
	// MetadataItems is number of metadata items allowed for each instance
	MetadataItems int `json:"metadata_items"`
	// Ram is megabytes allowed for each instance
	Ram int `json:"ram"`
	// SecurityGroupRules is rules allowed for each security group
	SecurityGroupRules int `json:"security_group_rules"`
	// SecurityGroups security groups allowed for each project
	SecurityGroups int `json:"security_groups"`
	// Cores is number of instance cores allowed for each project
	Cores int `json:"cores"`
	// Instances is number of instances allowed for each project
	Instances int `json:"instances"`
}

// QuotaSetPage stores a single, only page of QuotaSet results from a List call.
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

// Extract is a method that attempts to interpret any QuotaSet resource response as a QuotaSet struct.
func (r quotaResult) Extract() (*QuotaSet, error) {
	var s struct {
		QuotaSet *QuotaSet `json:"quota_set"`
	}
	err := r.ExtractInto(&s)
	return s.QuotaSet, err
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a QuotaSet.
type GetResult struct {
	quotaResult
}
