package clusters

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// CreateResult is the response of a Create operations.
type CreateResult struct {
	commonResult
}

// DeleteResult is the result from a Delete operation. Call its Extract or ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// Extract is a function that accepts a result and extracts a cluster resource.
func (r commonResult) Extract() (*Cluster, error) {
	var s *Cluster
	err := r.ExtractInto(&s)
	return s, err
}

// UpdateResult is the response of a Update operations.
type UpdateResult struct {
	commonResult
}

// ResizeResult is the response of a Resize operations.
type ResizeResult struct {
	commonResult
}

func (r CreateResult) Extract() (string, error) {
	var s struct {
		UUID string
	}
	err := r.ExtractInto(&s)
	return s.UUID, err
}

func (r UpdateResult) Extract() (string, error) {
	var s struct {
		UUID string
	}
	err := r.ExtractInto(&s)
	return s.UUID, err
}

type Cluster struct {
	APIAddress        string             `json:"api_address"`
	COEVersion        string             `json:"coe_version"`
	ClusterTemplateID string             `json:"cluster_template_id"`
	ContainerVersion  string             `json:"container_version"`
	CreateTimeout     int                `json:"create_timeout"`
	CreatedAt         time.Time          `json:"created_at"`
	DiscoveryURL      string             `json:"discovery_url"`
	DockerVolumeSize  int                `json:"docker_volume_size"`
	Faults            map[string]string  `json:"faults"`
	FlavorID          string             `json:"flavor_id"`
	KeyPair           string             `json:"keypair"`
	Labels            map[string]string  `json:"labels"`
	Links             []gophercloud.Link `json:"links"`
	MasterFlavorID    string             `json:"master_flavor_id"`
	MasterAddresses   []string           `json:"master_addresses"`
	MasterCount       int                `json:"master_count"`
	Name              string             `json:"name"`
	NodeAddresses     []string           `json:"node_addresses"`
	NodeCount         int                `json:"node_count"`
	ProjectID         string             `json:"project_id"`
	StackID           string             `json:"stack_id"`
	Status            string             `json:"status"`
	StatusReason      string             `json:"status_reason"`
	UUID              string             `json:"uuid"`
	UpdatedAt         time.Time          `json:"updated_at"`
	UserID            string             `json:"user_id"`
}

type ClusterPage struct {
	pagination.LinkedPageBase
}

func (r ClusterPage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, nil
}

// IsEmpty checks whether a ClusterPage struct is empty.
func (r ClusterPage) IsEmpty() (bool, error) {
	is, err := ExtractClusters(r)
	return len(is) == 0, err
}

func ExtractClusters(r pagination.Page) ([]Cluster, error) {
	var s struct {
		Clusters []Cluster `json:"clusters"`
	}
	err := (r.(ClusterPage)).ExtractInto(&s)
	return s.Clusters, err
}
