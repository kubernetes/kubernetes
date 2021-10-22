package limits

import (
	"github.com/gophercloud/gophercloud"
)

// Limits is a struct that contains the response of a limit query.
type Limits struct {
	// Absolute contains the limits and usage information.
	Absolute Absolute `json:"absolute"`
}

// Usage is a struct that contains the current resource usage and limits
// of a tenant.
type Absolute struct {
	// MaxTotalCores is the number of cores available to a tenant.
	MaxTotalCores int `json:"maxTotalCores"`

	// MaxImageMeta is the amount of image metadata available to a tenant.
	MaxImageMeta int `json:"maxImageMeta"`

	// MaxServerMeta is the amount of server metadata available to a tenant.
	MaxServerMeta int `json:"maxServerMeta"`

	// MaxPersonality is the amount of personality/files available to a tenant.
	MaxPersonality int `json:"maxPersonality"`

	// MaxPersonalitySize is the personality file size available to a tenant.
	MaxPersonalitySize int `json:"maxPersonalitySize"`

	// MaxTotalKeypairs is the total keypairs available to a tenant.
	MaxTotalKeypairs int `json:"maxTotalKeypairs"`

	// MaxSecurityGroups is the number of security groups available to a tenant.
	MaxSecurityGroups int `json:"maxSecurityGroups"`

	// MaxSecurityGroupRules is the number of security group rules available to
	// a tenant.
	MaxSecurityGroupRules int `json:"maxSecurityGroupRules"`

	// MaxServerGroups is the number of server groups available to a tenant.
	MaxServerGroups int `json:"maxServerGroups"`

	// MaxServerGroupMembers is the number of server group members available
	// to a tenant.
	MaxServerGroupMembers int `json:"maxServerGroupMembers"`

	// MaxTotalFloatingIps is the number of floating IPs available to a tenant.
	MaxTotalFloatingIps int `json:"maxTotalFloatingIps"`

	// MaxTotalInstances is the number of instances/servers available to a tenant.
	MaxTotalInstances int `json:"maxTotalInstances"`

	// MaxTotalRAMSize is the total amount of RAM available to a tenant measured
	// in megabytes (MB).
	MaxTotalRAMSize int `json:"maxTotalRAMSize"`

	// TotalCoresUsed is the number of cores currently in use.
	TotalCoresUsed int `json:"totalCoresUsed"`

	// TotalInstancesUsed is the number of instances/servers in use.
	TotalInstancesUsed int `json:"totalInstancesUsed"`

	// TotalFloatingIpsUsed is the number of floating IPs in use.
	TotalFloatingIpsUsed int `json:"totalFloatingIpsUsed"`

	// TotalRAMUsed is the total RAM/memory in use measured in megabytes (MB).
	TotalRAMUsed int `json:"totalRAMUsed"`

	// TotalSecurityGroupsUsed is the total number of security groups in use.
	TotalSecurityGroupsUsed int `json:"totalSecurityGroupsUsed"`

	// TotalServerGroupsUsed is the total number of server groups in use.
	TotalServerGroupsUsed int `json:"totalServerGroupsUsed"`
}

// Extract interprets a limits result as a Limits.
func (r GetResult) Extract() (*Limits, error) {
	var s struct {
		Limits *Limits `json:"limits"`
	}
	err := r.ExtractInto(&s)
	return s.Limits, err
}

// GetResult is the response from a Get operation. Call its Extract
// method to interpret it as an Absolute.
type GetResult struct {
	gophercloud.Result
}
