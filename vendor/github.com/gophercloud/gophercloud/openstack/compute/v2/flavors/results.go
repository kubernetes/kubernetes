package flavors

import (
	"encoding/json"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// GetResult temporarily holds the response from a Get call.
type GetResult struct {
	gophercloud.Result
}

// Extract provides access to the individual Flavor returned by the Get function.
func (r GetResult) Extract() (*Flavor, error) {
	var s struct {
		Flavor *Flavor `json:"flavor"`
	}
	err := r.ExtractInto(&s)
	return s.Flavor, err
}

// Flavor records represent (virtual) hardware configurations for server resources in a region.
type Flavor struct {
	// The Id field contains the flavor's unique identifier.
	// For example, this identifier will be useful when specifying which hardware configuration to use for a new server instance.
	ID string `json:"id"`
	// The Disk and RA< fields provide a measure of storage space offered by the flavor, in GB and MB, respectively.
	Disk int `json:"disk"`
	RAM  int `json:"ram"`
	// The Name field provides a human-readable moniker for the flavor.
	Name       string  `json:"name"`
	RxTxFactor float64 `json:"rxtx_factor"`
	// Swap indicates how much space is reserved for swap.
	// If not provided, this field will be set to 0.
	Swap int `json:"swap"`
	// VCPUs indicates how many (virtual) CPUs are available for this flavor.
	VCPUs int `json:"vcpus"`
}

func (f *Flavor) UnmarshalJSON(b []byte) error {
	var flavor struct {
		ID         string      `json:"id"`
		Disk       int         `json:"disk"`
		RAM        int         `json:"ram"`
		Name       string      `json:"name"`
		RxTxFactor float64     `json:"rxtx_factor"`
		Swap       interface{} `json:"swap"`
		VCPUs      int         `json:"vcpus"`
	}
	err := json.Unmarshal(b, &flavor)
	if err != nil {
		return err
	}

	f.ID = flavor.ID
	f.Disk = flavor.Disk
	f.RAM = flavor.RAM
	f.Name = flavor.Name
	f.RxTxFactor = flavor.RxTxFactor
	f.VCPUs = flavor.VCPUs

	switch t := flavor.Swap.(type) {
	case float64:
		f.Swap = int(t)
	case string:
		switch t {
		case "":
			f.Swap = 0
		default:
			swap, err := strconv.ParseFloat(t, 64)
			if err != nil {
				return err
			}
			f.Swap = int(swap)
		}
	}

	return nil
}

// FlavorPage contains a single page of the response from a List call.
type FlavorPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a page contains any results.
func (page FlavorPage) IsEmpty() (bool, error) {
	flavors, err := ExtractFlavors(page)
	return len(flavors) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the next page of results.
func (page FlavorPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"flavors_links"`
	}
	err := page.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractFlavors provides access to the list of flavors in a page acquired from the List operation.
func ExtractFlavors(r pagination.Page) ([]Flavor, error) {
	var s struct {
		Flavors []Flavor `json:"flavors"`
	}
	err := (r.(FlavorPage)).ExtractInto(&s)
	return s.Flavors, err
}
