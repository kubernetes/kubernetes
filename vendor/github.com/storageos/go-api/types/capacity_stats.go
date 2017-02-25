package types

// ErrCapacityStatsUnchanged can be used when comparing stats
const ErrCapacityStatsUnchanged = "no changes"

// CapacityStats is used to report capacity statistics on pools and controllers.
type CapacityStats struct {

	// TotalCapacityBytes is the object's total capacity in bytes.
	TotalCapacityBytes uint64 `json:"total_capacity_bytes"`

	// AvailableCapacityBytes is the object's available capacity in bytes.
	AvailableCapacityBytes uint64 `json:"available_capacity_bytes"`

	// ProvisionedCapacityBytes is the object's provisioned capacity in bytes.
	ProvisionedCapacityBytes uint64 `json:"provisioned_capacity_bytes"`
}

// IsEqual checks if capacity values are the same
func (c CapacityStats) IsEqual(n CapacityStats) bool {
	if c == n {
		return true
	}
	return false
}
