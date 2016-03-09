package flavors

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

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

	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		WeaklyTypedInput: true,
		Result:           &result,
	})

	err = decoder.Decode(gr.Body)
	return &result.Flavor, err
}

// Flavor records represent (virtual) hardware configurations for server resources in a region.
type Flavor struct {
	// The flavor's unique identifier.
	ID string `mapstructure:"id"`

	// The RAM capacity for the flavor.
	RAM int `mapstructure:"ram"`

	// The Name field provides a human-readable moniker for the flavor.
	Name string `mapstructure:"name"`

	// Links to access the flavor.
	Links []gophercloud.Link
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

// ExtractFlavors provides access to the list of flavors in a page acquired from the List operation.
func ExtractFlavors(page pagination.Page) ([]Flavor, error) {
	casted := page.(FlavorPage).Body
	var container struct {
		Flavors []Flavor `mapstructure:"flavors"`
	}

	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		WeaklyTypedInput: true,
		Result:           &container,
	})

	err = decoder.Decode(casted)

	return container.Flavors, err
}
