package instances

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/openstack/db/v1/flavors"
	os "github.com/rackspace/gophercloud/openstack/db/v1/instances"
	"github.com/rackspace/gophercloud/pagination"
)

// Instance represents a remote MySQL instance.
type Instance struct {
	// Indicates the datetime that the instance was created
	Created time.Time `mapstructure:"-"`

	// Indicates the most recent datetime that the instance was updated.
	Updated time.Time `mapstructure:"-"`

	// Indicates how the instance stores data.
	Datastore datastores.DatastorePartial

	// Indicates the hardware flavor the instance uses.
	Flavor flavors.Flavor

	// A DNS-resolvable hostname associated with the database instance (rather
	// than an IPv4 address). Since the hostname always resolves to the correct
	// IP address of the database instance, this relieves the user from the task
	// of maintaining the mapping. Note that although the IP address may likely
	// change on resizing, migrating, and so forth, the hostname always resolves
	// to the correct database instance.
	Hostname string

	// Indicates the unique identifier for the instance resource.
	ID string

	// Exposes various links that reference the instance resource.
	Links []gophercloud.Link

	// The human-readable name of the instance.
	Name string

	// The build status of the instance.
	Status string

	// Information about the attached volume of the instance.
	Volume os.Volume

	// IP indicates the various IP addresses which allow access.
	IP []string

	// Indicates whether this instance is a replica of another source instance.
	ReplicaOf *Instance `mapstructure:"replica_of" json:"replica_of"`

	// Indicates whether this instance is the source of other replica instances.
	Replicas []Instance
}

func commonExtract(err error, body interface{}) (*Instance, error) {
	if err != nil {
		return nil, err
	}

	var response struct {
		Instance Instance `mapstructure:"instance"`
	}

	err = mapstructure.Decode(body, &response)

	val := body.(map[string]interface{})["instance"].(map[string]interface{})

	if t, ok := val["created"].(string); ok && t != "" {
		creationTime, err := time.Parse(time.RFC3339, t)
		if err != nil {
			return &response.Instance, err
		}
		response.Instance.Created = creationTime
	}

	if t, ok := val["updated"].(string); ok && t != "" {
		updatedTime, err := time.Parse(time.RFC3339, t)
		if err != nil {
			return &response.Instance, err
		}
		response.Instance.Updated = updatedTime
	}

	return &response.Instance, err
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	os.CreateResult
}

// Extract will retrieve an instance from a create result.
func (r CreateResult) Extract() (*Instance, error) {
	return commonExtract(r.Err, r.Body)
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	os.GetResult
}

// Extract will extract an Instance from a GetResult.
func (r GetResult) Extract() (*Instance, error) {
	return commonExtract(r.Err, r.Body)
}

// ConfigResult represents the result of getting default configuration for an
// instance.
type ConfigResult struct {
	gophercloud.Result
}

// DetachResult represents the result of detaching a replica from its source.
type DetachResult struct {
	gophercloud.ErrResult
}

// Extract will extract the configuration information (in the form of a map)
// about a particular instance.
func (r ConfigResult) Extract() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Instance struct {
			Config map[string]string `mapstructure:"configuration"`
		} `mapstructure:"instance"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return response.Instance.Config, err
}

// UpdateResult represents the result of an Update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// ExtractInstances retrieves a slice of instances from a paginated collection.
func ExtractInstances(page pagination.Page) ([]Instance, error) {
	casted := page.(os.InstancePage).Body

	var resp struct {
		Instances []Instance `mapstructure:"instances"`
	}

	if err := mapstructure.Decode(casted, &resp); err != nil {
		return nil, err
	}

	var vals []interface{}
	switch casted.(type) {
	case map[string]interface{}:
		vals = casted.(map[string]interface{})["instances"].([]interface{})
	case map[string][]interface{}:
		vals = casted.(map[string][]interface{})["instances"]
	default:
		return resp.Instances, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i, v := range vals {
		val := v.(map[string]interface{})

		if t, ok := val["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return resp.Instances, err
			}
			resp.Instances[i].Created = creationTime
		}

		if t, ok := val["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return resp.Instances, err
			}
			resp.Instances[i].Updated = updatedTime
		}
	}

	return resp.Instances, nil
}
