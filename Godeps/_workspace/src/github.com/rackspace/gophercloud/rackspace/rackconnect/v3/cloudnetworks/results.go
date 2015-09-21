package cloudnetworks

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// CloudNetwork represents a network associated with a RackConnect configuration.
type CloudNetwork struct {
	// Specifies the ID of the newtork.
	ID string `mapstructure:"id"`
	// Specifies the user-provided name of the network.
	Name string `mapstructure:"name"`
	// Specifies the IP range for this network.
	CIDR string `mapstructure:"cidr"`
	// Specifies the time the network was created.
	CreatedAt time.Time `mapstructure:"-"`
	// Specifies the time the network was last updated.
	UpdatedAt time.Time `mapstructure:"-"`
}

// CloudNetworkPage is the page returned by a pager when traversing over a
// collection of CloudNetworks.
type CloudNetworkPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a CloudNetworkPage contains no CloudNetworks.
func (r CloudNetworkPage) IsEmpty() (bool, error) {
	cns, err := ExtractCloudNetworks(r)
	if err != nil {
		return true, err
	}
	return len(cns) == 0, nil
}

// ExtractCloudNetworks extracts and returns CloudNetworks. It is used while iterating over
// a cloudnetworks.List call.
func ExtractCloudNetworks(page pagination.Page) ([]CloudNetwork, error) {
	var res []CloudNetwork
	casted := page.(CloudNetworkPage).Body
	err := mapstructure.Decode(casted, &res)

	var rawNets []interface{}
	switch casted.(type) {
	case interface{}:
		rawNets = casted.([]interface{})
	default:
		return res, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawNets {
		thisNet := (rawNets[i]).(map[string]interface{})

		if t, ok := thisNet["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].CreatedAt = creationTime
		}

		if t, ok := thisNet["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].UpdatedAt = updatedTime
		}
	}

	return res, err
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that extracts a CloudNetwork from a GetResult.
func (r GetResult) Extract() (*CloudNetwork, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res CloudNetwork

	err := mapstructure.Decode(r.Body, &res)

	b := r.Body.(map[string]interface{})

	if date, ok := b["created"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.CreatedAt = t
	}

	if date, ok := b["updated"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.UpdatedAt = t
	}

	return &res, err
}
