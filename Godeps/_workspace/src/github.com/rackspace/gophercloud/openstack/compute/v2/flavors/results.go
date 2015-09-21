package flavors

import (
	"errors"
	"reflect"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ErrCannotInterpret is returned by an Extract call if the response body doesn't have the expected structure.
var ErrCannotInterpet = errors.New("Unable to interpret a response body.")

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
}

// FlavorPage contains a single page of the response from a List call.
type FlavorPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a page contains any results.
func (p FlavorPage) IsEmpty() (bool, error) {
	flavors, err := ExtractFlavors(p)
	if err != nil {
		return true, err
	}
	return len(flavors) == 0, nil
}

// NextPageURL uses the response's embedded link reference to navigate to the next page of results.
func (p FlavorPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"flavors_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

func defaulter(from, to reflect.Kind, v interface{}) (interface{}, error) {
	if (from == reflect.String) && (to == reflect.Int) {
		return 0, nil
	}
	return v, nil
}

// ExtractFlavors provides access to the list of flavors in a page acquired from the List operation.
func ExtractFlavors(page pagination.Page) ([]Flavor, error) {
	casted := page.(FlavorPage).Body
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
