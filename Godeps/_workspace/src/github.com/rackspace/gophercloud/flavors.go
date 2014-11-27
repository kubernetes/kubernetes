package gophercloud

import (
	"github.com/racker/perigee"
)

// See CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListFlavors() ([]Flavor, error) {
	var fs []Flavor

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/flavors/detail"
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results:      &struct{ Flavors *[]Flavor }{&fs},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return fs, err
}

// FlavorLink provides a reference to a flavor by either ID or by direct URL.
// Some services use just the ID, others use just the URL.
// This structure provides a common means of expressing both in a single field.
type FlavorLink struct {
	Id    string `json:"id"`
	Links []Link `json:"links"`
}

// Flavor records represent (virtual) hardware configurations for server resources in a region.
//
// The Id field contains the flavor's unique identifier.
// For example, this identifier will be useful when specifying which hardware configuration to use for a new server instance.
//
// The Disk and Ram fields provide a measure of storage space offered by the flavor, in GB and MB, respectively.
//
// The Name field provides a human-readable moniker for the flavor.
//
// Swap indicates how much space is reserved for swap.
// If not provided, this field will be set to 0.
//
// VCpus indicates how many (virtual) CPUs are available for this flavor.
type Flavor struct {
	OsFlvDisabled bool    `json:"OS-FLV-DISABLED:disabled"`
	Disk          int     `json:"disk"`
	Id            string  `json:"id"`
	Links         []Link  `json:"links"`
	Name          string  `json:"name"`
	Ram           int     `json:"ram"`
	RxTxFactor    float64 `json:"rxtx_factor"`
	Swap          int     `json:"swap"`
	VCpus         int     `json:"vcpus"`
}
