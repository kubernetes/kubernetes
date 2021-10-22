package networkipavailabilities

import (
	"encoding/json"
	"math/big"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// GetResult represents the result of a Get operation. Call its Extract
// method to interpret it as a NetworkIPAvailability.
type GetResult struct {
	commonResult
}

// Extract is a function that accepts a result and extracts a NetworkIPAvailability.
func (r commonResult) Extract() (*NetworkIPAvailability, error) {
	var s struct {
		NetworkIPAvailability *NetworkIPAvailability `json:"network_ip_availability"`
	}
	err := r.ExtractInto(&s)
	return s.NetworkIPAvailability, err
}

// NetworkIPAvailability represents availability details for a single network.
type NetworkIPAvailability struct {
	// NetworkID contains an unique identifier of the network.
	NetworkID string `json:"network_id"`

	// NetworkName represents human-readable name of the network.
	NetworkName string `json:"network_name"`

	// ProjectID is the ID of the Identity project.
	ProjectID string `json:"project_id"`

	// TenantID is the ID of the Identity project.
	TenantID string `json:"tenant_id"`

	// SubnetIPAvailabilities contains availability details for every subnet
	// that is associated to the network.
	SubnetIPAvailabilities []SubnetIPAvailability `json:"subnet_ip_availability"`

	// TotalIPs represents a number of IP addresses in the network.
	TotalIPs string `json:"-"`

	// UsedIPs represents a number of used IP addresses in the network.
	UsedIPs string `json:"-"`
}

func (r *NetworkIPAvailability) UnmarshalJSON(b []byte) error {
	type tmp NetworkIPAvailability
	var s struct {
		tmp
		TotalIPs big.Int `json:"total_ips"`
		UsedIPs  big.Int `json:"used_ips"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = NetworkIPAvailability(s.tmp)

	r.TotalIPs = s.TotalIPs.String()
	r.UsedIPs = s.UsedIPs.String()

	return err
}

// SubnetIPAvailability represents availability details for a single subnet.
type SubnetIPAvailability struct {
	// SubnetID contains an unique identifier of the subnet.
	SubnetID string `json:"subnet_id"`

	// SubnetName represents human-readable name of the subnet.
	SubnetName string `json:"subnet_name"`

	// CIDR represents prefix in the CIDR format.
	CIDR string `json:"cidr"`

	// IPVersion is the IP protocol version.
	IPVersion int `json:"ip_version"`

	// TotalIPs represents a number of IP addresses in the subnet.
	TotalIPs string `json:"-"`

	// UsedIPs represents a number of used IP addresses in the subnet.
	UsedIPs string `json:"-"`
}

func (r *SubnetIPAvailability) UnmarshalJSON(b []byte) error {
	type tmp SubnetIPAvailability
	var s struct {
		tmp
		TotalIPs big.Int `json:"total_ips"`
		UsedIPs  big.Int `json:"used_ips"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = SubnetIPAvailability(s.tmp)

	r.TotalIPs = s.TotalIPs.String()
	r.UsedIPs = s.UsedIPs.String()

	return err
}

// NetworkIPAvailabilityPage stores a single page of NetworkIPAvailabilities
// from the List call.
type NetworkIPAvailabilityPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a NetworkIPAvailability is empty.
func (r NetworkIPAvailabilityPage) IsEmpty() (bool, error) {
	networkipavailabilities, err := ExtractNetworkIPAvailabilities(r)
	return len(networkipavailabilities) == 0, err
}

// ExtractNetworkIPAvailabilities interprets the results of a single page from
// a List() API call, producing a slice of NetworkIPAvailabilities structures.
func ExtractNetworkIPAvailabilities(r pagination.Page) ([]NetworkIPAvailability, error) {
	var s struct {
		NetworkIPAvailabilities []NetworkIPAvailability `json:"network_ip_availabilities"`
	}
	err := (r.(NetworkIPAvailabilityPage)).ExtractInto(&s)
	if err != nil {
		return nil, err
	}
	return s.NetworkIPAvailabilities, nil
}
