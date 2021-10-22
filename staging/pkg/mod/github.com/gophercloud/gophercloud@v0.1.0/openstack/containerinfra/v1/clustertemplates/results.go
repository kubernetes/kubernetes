package clustertemplates

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

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult is the response of a Get operations.
type GetResult struct {
	commonResult
}

// UpdateResult is the response of a Update operations.
type UpdateResult struct {
	commonResult
}

// Extract is a function that accepts a result and extracts a cluster-template resource.
func (r commonResult) Extract() (*ClusterTemplate, error) {
	var s *ClusterTemplate
	err := r.ExtractInto(&s)
	return s, err
}

// Represents a template for a Cluster Template
type ClusterTemplate struct {
	APIServerPort       int                `json:"apiserver_port"`
	COE                 string             `json:"coe"`
	ClusterDistro       string             `json:"cluster_distro"`
	CreatedAt           time.Time          `json:"created_at"`
	DNSNameServer       string             `json:"dns_nameserver"`
	DockerStorageDriver string             `json:"docker_storage_driver"`
	DockerVolumeSize    int                `json:"docker_volume_size"`
	ExternalNetworkID   string             `json:"external_network_id"`
	FixedNetwork        string             `json:"fixed_network"`
	FixedSubnet         string             `json:"fixed_subnet"`
	FlavorID            string             `json:"flavor_id"`
	FloatingIPEnabled   bool               `json:"floating_ip_enabled"`
	HTTPProxy           string             `json:"http_proxy"`
	HTTPSProxy          string             `json:"https_proxy"`
	ImageID             string             `json:"image_id"`
	InsecureRegistry    string             `json:"insecure_registry"`
	KeyPairID           string             `json:"keypair_id"`
	Labels              map[string]string  `json:"labels"`
	Links               []gophercloud.Link `json:"links"`
	MasterFlavorID      string             `json:"master_flavor_id"`
	MasterLBEnabled     bool               `json:"master_lb_enabled"`
	Name                string             `json:"name"`
	NetworkDriver       string             `json:"network_driver"`
	NoProxy             string             `json:"no_proxy"`
	ProjectID           string             `json:"project_id"`
	Public              bool               `json:"public"`
	RegistryEnabled     bool               `json:"registry_enabled"`
	ServerType          string             `json:"server_type"`
	TLSDisabled         bool               `json:"tls_disabled"`
	UUID                string             `json:"uuid"`
	UpdatedAt           time.Time          `json:"updated_at"`
	UserID              string             `json:"user_id"`
	VolumeDriver        string             `json:"volume_driver"`
}

// ClusterTemplatePage is the page returned by a pager when traversing over a
// collection of cluster-templates.
type ClusterTemplatePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of cluster template has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r ClusterTemplatePage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, nil
}

// IsEmpty checks whether a ClusterTemplatePage struct is empty.
func (r ClusterTemplatePage) IsEmpty() (bool, error) {
	is, err := ExtractClusterTemplates(r)
	return len(is) == 0, err
}

// ExtractClusterTemplates accepts a Page struct, specifically a ClusterTemplatePage struct,
// and extracts the elements into a slice of cluster templates structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractClusterTemplates(r pagination.Page) ([]ClusterTemplate, error) {
	var s struct {
		ClusterTemplates []ClusterTemplate `json:"clustertemplates"`
	}
	err := (r.(ClusterTemplatePage)).ExtractInto(&s)
	return s.ClusterTemplates, err
}
