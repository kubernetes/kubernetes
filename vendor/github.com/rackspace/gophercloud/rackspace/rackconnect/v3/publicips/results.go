package publicips

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// PublicIP represents a public IP address.
type PublicIP struct {
	// The unique ID of the public IP.
	ID string `mapstructure:"id"`
	// The IPv4 address of the public IP.
	PublicIPv4 string `mapstructure:"public_ip_v4"`
	// The cloud server (node) of the public IP.
	CloudServer struct {
		// The cloud server ID.
		ID string `mapstructure:"id"`
		// The name of the server.
		Name string `mapstructure:"name"`
		// The cloud network for the cloud server.
		CloudNetwork struct {
			// The network ID.
			ID string `mapstructure:"id"`
			// The network name.
			Name string `mapstructure:"name"`
			// The network's private IPv4 address.
			PrivateIPv4 string `mapstructure:"private_ip_v4"`
			// The IP range for the network.
			CIDR string `mapstructure:"cidr"`
			// The datetime the network was created.
			CreatedAt time.Time `mapstructure:"-"`
			// The last datetime the network was updated.
			UpdatedAt time.Time `mapstructure:"-"`
		} `mapstructure:"cloud_network"`
		// The datetime the server was created.
		CreatedAt time.Time `mapstructure:"-"`
		// The datetime the server was last updated.
		UpdatedAt time.Time `mapstructure:"-"`
	} `mapstructure:"cloud_server"`
	// The status of the public IP.
	Status string `mapstructure:"status"`
	// The details of the status of the public IP.
	StatusDetail string `mapstructure:"status_detail"`
	// The time the public IP was created.
	CreatedAt time.Time `mapstructure:"-"`
	// The time the public IP was last updated.
	UpdatedAt time.Time `mapstructure:"-"`
}

// PublicIPPage is the page returned by a pager when traversing over a
// collection of PublicIPs.
type PublicIPPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a PublicIPPage contains no PublicIPs.
func (r PublicIPPage) IsEmpty() (bool, error) {
	n, err := ExtractPublicIPs(r)
	if err != nil {
		return true, err
	}
	return len(n) == 0, nil
}

// ExtractPublicIPs extracts and returns a slice of PublicIPs. It is used while iterating over
// a publicips.List call.
func ExtractPublicIPs(page pagination.Page) ([]PublicIP, error) {
	var res []PublicIP
	casted := page.(PublicIPPage).Body
	err := mapstructure.Decode(casted, &res)

	var rawNodesDetails []interface{}
	switch casted.(type) {
	case interface{}:
		rawNodesDetails = casted.([]interface{})
	default:
		return res, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawNodesDetails {
		thisNodeDetails := (rawNodesDetails[i]).(map[string]interface{})

		if t, ok := thisNodeDetails["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].CreatedAt = creationTime
		}

		if t, ok := thisNodeDetails["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].UpdatedAt = updatedTime
		}

		if cs, ok := thisNodeDetails["cloud_server"].(map[string]interface{}); ok {
			if t, ok := cs["created"].(string); ok && t != "" {
				creationTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return res, err
				}
				res[i].CloudServer.CreatedAt = creationTime
			}
			if t, ok := cs["updated"].(string); ok && t != "" {
				updatedTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return res, err
				}
				res[i].CloudServer.UpdatedAt = updatedTime
			}
			if cn, ok := cs["cloud_network"].(map[string]interface{}); ok {
				if t, ok := cn["created"].(string); ok && t != "" {
					creationTime, err := time.Parse(time.RFC3339, t)
					if err != nil {
						return res, err
					}
					res[i].CloudServer.CloudNetwork.CreatedAt = creationTime
				}
				if t, ok := cn["updated"].(string); ok && t != "" {
					updatedTime, err := time.Parse(time.RFC3339, t)
					if err != nil {
						return res, err
					}
					res[i].CloudServer.CloudNetwork.UpdatedAt = updatedTime
				}
			}
		}
	}

	return res, err
}

// PublicIPResult represents a result that can be extracted into a PublicIP.
type PublicIPResult struct {
	gophercloud.Result
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	PublicIPResult
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	PublicIPResult
}

// Extract is a function that extracts a PublicIP from a PublicIPResult.
func (r PublicIPResult) Extract() (*PublicIP, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res PublicIP
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

	if cs, ok := b["cloud_server"].(map[string]interface{}); ok {
		if t, ok := cs["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return &res, err
			}
			res.CloudServer.CreatedAt = creationTime
		}
		if t, ok := cs["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return &res, err
			}
			res.CloudServer.UpdatedAt = updatedTime
		}
		if cn, ok := cs["cloud_network"].(map[string]interface{}); ok {
			if t, ok := cn["created"].(string); ok && t != "" {
				creationTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return &res, err
				}
				res.CloudServer.CloudNetwork.CreatedAt = creationTime
			}
			if t, ok := cn["updated"].(string); ok && t != "" {
				updatedTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return &res, err
				}
				res.CloudServer.CloudNetwork.UpdatedAt = updatedTime
			}
		}
	}

	return &res, err
}

// DeleteResult represents the result of a Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
