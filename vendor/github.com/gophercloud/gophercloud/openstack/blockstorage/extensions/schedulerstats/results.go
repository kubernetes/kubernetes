package schedulerstats

import (
	"encoding/json"
	"math"

	"github.com/gophercloud/gophercloud/pagination"
)

// Capabilities represents the information of an individual StoragePool.
type Capabilities struct {
	// The following fields should be present in all storage drivers.
	DriverVersion     string  `json:"driver_version"`
	FreeCapacityGB    float64 `json:"-"`
	StorageProtocol   string  `json:"storage_protocol"`
	TotalCapacityGB   float64 `json:"-"`
	VendorName        string  `json:"vendor_name"`
	VolumeBackendName string  `json:"volume_backend_name"`

	// The following fields are optional and may have empty values depending
	// on the storage driver in use.
	ReservedPercentage       int64   `json:"reserved_percentage"`
	LocationInfo             string  `json:"location_info"`
	QoSSupport               bool    `json:"QoS_support"`
	ProvisionedCapacityGB    float64 `json:"provisioned_capacity_gb"`
	MaxOverSubscriptionRatio float64 `json:"max_over_subscription_ratio"`
	ThinProvisioningSupport  bool    `json:"thin_provisioning_support"`
	ThickProvisioningSupport bool    `json:"thick_provisioning_support"`
	TotalVolumes             int64   `json:"total_volumes"`
	FilterFunction           string  `json:"filter_function"`
	GoodnessFuction          string  `json:"goodness_function"`
	Mutliattach              bool    `json:"multiattach"`
	SparseCopyVolume         bool    `json:"sparse_copy_volume"`
}

// StoragePool represents an individual StoragePool retrieved from the
// schedulerstats API.
type StoragePool struct {
	Name         string       `json:"name"`
	Capabilities Capabilities `json:"capabilities"`
}

func (r *Capabilities) UnmarshalJSON(b []byte) error {
	type tmp Capabilities
	var s struct {
		tmp
		FreeCapacityGB  interface{} `json:"free_capacity_gb"`
		TotalCapacityGB interface{} `json:"total_capacity_gb"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Capabilities(s.tmp)

	// Generic function to parse a capacity value which may be a numeric
	// value, "unknown", or "infinite"
	parseCapacity := func(capacity interface{}) float64 {
		if capacity != nil {
			switch capacity.(type) {
			case float64:
				return capacity.(float64)
			case string:
				if capacity.(string) == "infinite" {
					return math.Inf(1)
				}
			}
		}
		return 0.0
	}

	r.FreeCapacityGB = parseCapacity(s.FreeCapacityGB)
	r.TotalCapacityGB = parseCapacity(s.TotalCapacityGB)

	return nil
}

// StoragePoolPage is a single page of all List results.
type StoragePoolPage struct {
	pagination.SinglePageBase
}

// IsEmpty satisfies the IsEmpty method of the Page interface. It returns true
// if a List contains no results.
func (page StoragePoolPage) IsEmpty() (bool, error) {
	va, err := ExtractStoragePools(page)
	return len(va) == 0, err
}

// ExtractStoragePools takes a List result and extracts the collection of
// StoragePools returned by the API.
func ExtractStoragePools(p pagination.Page) ([]StoragePool, error) {
	var s struct {
		StoragePools []StoragePool `json:"pools"`
	}
	err := (p.(StoragePoolPage)).ExtractInto(&s)
	return s.StoragePools, err
}
