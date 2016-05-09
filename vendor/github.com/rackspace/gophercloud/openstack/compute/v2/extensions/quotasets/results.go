package quotasets

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// QuotaSet is a set of operational limits that allow for control of compute usage.
type QuotaSet struct {
	//ID is tenant associated with this quota_set
	ID string `mapstructure:"id"`
	//FixedIps is number of fixed ips alloted this quota_set
	FixedIps int `mapstructure:"fixed_ips"`
	// FloatingIps is number of floating ips alloted this quota_set
	FloatingIps int `mapstructure:"floating_ips"`
	// InjectedFileContentBytes is content bytes allowed for each injected file
	InjectedFileContentBytes int `mapstructure:"injected_file_content_bytes"`
	// InjectedFilePathBytes is allowed bytes for each injected file path
	InjectedFilePathBytes int `mapstructure:"injected_file_path_bytes"`
	// InjectedFiles is injected files allowed for each project
	InjectedFiles int `mapstructure:"injected_files"`
	// KeyPairs is number of ssh keypairs
	KeyPairs int `mapstructure:"keypairs"`
	// MetadataItems is number of metadata items allowed for each instance
	MetadataItems int `mapstructure:"metadata_items"`
	// Ram is megabytes allowed for each instance
	Ram int `mapstructure:"ram"`
	// SecurityGroupRules is rules allowed for each security group
	SecurityGroupRules int `mapstructure:"security_group_rules"`
	// SecurityGroups security groups allowed for each project
	SecurityGroups int `mapstructure:"security_groups"`
	// Cores is number of instance cores allowed for each project
	Cores int `mapstructure:"cores"`
	// Instances is number of instances allowed for each project
	Instances int `mapstructure:"instances"`
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
func ExtractQuotaSets(page pagination.Page) ([]QuotaSet, error) {
	var resp struct {
		QuotaSets []QuotaSet `mapstructure:"quotas"`
	}

	err := mapstructure.Decode(page.(QuotaSetPage).Body, &resp)
	results := make([]QuotaSet, len(resp.QuotaSets))
	for i, q := range resp.QuotaSets {
		results[i] = q
	}
	return results, err
}

type quotaResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any QuotaSet resource response as a QuotaSet struct.
func (r quotaResult) Extract() (*QuotaSet, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		QuotaSet *QuotaSet `json:"quota_set" mapstructure:"quota_set"`
	}

	err := mapstructure.Decode(r.Body, &res)
	return res.QuotaSet, err
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a QuotaSet.
type GetResult struct {
	quotaResult
}
