package flavors

import (
	"reflect"

	"github.com/rackspace/gophercloud"
	"github.com/mitchellh/mapstructure"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/flavors"
	"github.com/rackspace/gophercloud/pagination"
)

// ExtraSpecs provide additional information about the flavor.
type ExtraSpecs struct {
	// The number of data disks
	NumDataDisks int `mapstructure:"number_of_data_disks"`
	// The flavor class
	Class string `mapstructure:"class"`
	// Relative measure of disk I/O performance from 0-99, where higher is faster
	DiskIOIndex int    `mapstructure:"disk_io_index"`
	PolicyClass string `mapstructure:"policy_class"`
}

// Flavor records represent (virtual) hardware configurations for server resources in a region.
type Flavor struct {
	// The Id field contains the flavor's unique identifier.
	// For example, this identifier will be useful when specifying which hardware configuration to use for a new server instance.
	ID string `mapstructure:"id"`

	// The Disk and RA< fields provide a measure of storage space offered by the flavor, in GB and MB, respectively.
	Disk int `mapstructure:"disk"`
	RAM  int `mapstructure:"ram"`

	// The Name field provides a human-readable moniker for the flavor.
	Name string `mapstructure:"name"`

	RxTxFactor float64 `mapstructure:"rxtx_factor"`

	// Swap indicates how much space is reserved for swap.
	// If not provided, this field will be set to 0.
	Swap int `mapstructure:"swap"`

	// VCPUs indicates how many (virtual) CPUs are available for this flavor.
	VCPUs int `mapstructure:"vcpus"`

	// ExtraSpecs provides extra information about the flavor
	ExtraSpecs ExtraSpecs `mapstructure:"OS-FLV-WITH-EXT-SPECS:extra_specs"`
}

// GetResult temporarily holds the response from a Get call.
type GetResult struct {
	gophercloud.Result
}

// Extract provides access to the individual Flavor returned by the Get function.
func (gr GetResult) Extract() (*Flavor, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}

	var result struct {
		Flavor Flavor `mapstructure:"flavor"`
	}

	cfg := &mapstructure.DecoderConfig{
		DecodeHook: defaulter,
		Result:     &result,
	}
	decoder, err := mapstructure.NewDecoder(cfg)
	if err != nil {
		return nil, err
	}
	err = decoder.Decode(gr.Body)
	return &result.Flavor, err
}

func defaulter(from, to reflect.Kind, v interface{}) (interface{}, error) {
	if (from == reflect.String) && (to == reflect.Int) {
		return 0, nil
	}
	return v, nil
}

// ExtractFlavors provides access to the list of flavors in a page acquired from the List operation.
func ExtractFlavors(page pagination.Page) ([]Flavor, error) {
	casted := page.(os.FlavorPage).Body
	var container struct {
		Flavors []Flavor `mapstructure:"flavors"`
	}

	cfg := &mapstructure.DecoderConfig{
		DecodeHook: defaulter,
		Result:     &container,
	}
	decoder, err := mapstructure.NewDecoder(cfg)
	if err != nil {
		return container.Flavors, err
	}
	err = decoder.Decode(casted)
	if err != nil {
		return container.Flavors, err
	}

	return container.Flavors, nil
}
