package clusters

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Cluster represents an OpenStack Clustering cluster.
type Cluster struct {
	Config          map[string]interface{} `json:"config"`
	CreatedAt       time.Time              `json:"-"`
	Data            map[string]interface{} `json:"data"`
	Dependents      map[string]interface{} `json:"dependents"`
	DesiredCapacity int                    `json:"desired_capacity"`
	Domain          string                 `json:"domain"`
	ID              string                 `json:"id"`
	InitAt          time.Time              `json:"-"`
	MaxSize         int                    `json:"max_size"`
	Metadata        map[string]interface{} `json:"metadata"`
	MinSize         int                    `json:"min_size"`
	Name            string                 `json:"name"`
	Nodes           []string               `json:"nodes"`
	Policies        []string               `json:"policies"`
	ProfileID       string                 `json:"profile_id"`
	ProfileName     string                 `json:"profile_name"`
	Project         string                 `json:"project"`
	Status          string                 `json:"status"`
	StatusReason    string                 `json:"status_reason"`
	Timeout         int                    `json:"timeout"`
	UpdatedAt       time.Time              `json:"-"`
	User            string                 `json:"user"`
}

func (r *Cluster) UnmarshalJSON(b []byte) error {
	type tmp Cluster
	var s struct {
		tmp
		CreatedAt string `json:"created_at"`
		InitAt    string `json:"init_at"`
		UpdatedAt string `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Cluster(s.tmp)

	if s.CreatedAt != "" {
		r.CreatedAt, err = time.Parse(gophercloud.RFC3339Milli, s.CreatedAt)
		if err != nil {
			return err
		}
	}

	if s.InitAt != "" {
		r.InitAt, err = time.Parse(gophercloud.RFC3339Milli, s.InitAt)
		if err != nil {
			return err
		}
	}

	if s.UpdatedAt != "" {
		r.UpdatedAt, err = time.Parse(gophercloud.RFC3339Milli, s.UpdatedAt)
		if err != nil {
			return err
		}
	}

	return nil
}

// ClusterPolicy represents and OpenStack Clustering cluster policy.
type ClusterPolicy struct {
	ClusterID   string `json:"cluster_id"`
	ClusterName string `json:"cluster_name"`
	Enabled     bool   `json:"enabled"`
	ID          string `json:"id"`
	PolicyID    string `json:"policy_id"`
	PolicyName  string `json:"policy_name"`
	PolicyType  string `json:"policy_type"`
}

type ClusterAttributes struct {
	ID    string      `json:"id"`
	Value interface{} `json:"value"`
}

// Action represents an OpenStack Clustering action.
type Action struct {
	Action string `json:"action"`
}

// commonResult is the response of a base result.
type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult-based result as a Cluster.
func (r commonResult) Extract() (*Cluster, error) {
	var s struct {
		Cluster *Cluster `json:"cluster"`
	}

	err := r.ExtractInto(&s)
	return s.Cluster, err
}

// CreateResult is the response of a Create operations. Call its Extract method
// to interpret it as a Cluster.
type CreateResult struct {
	commonResult
}

// GetResult is the response of a Get operations. Call its Extract method to
// interpret it as a Cluster.
type GetResult struct {
	commonResult
}

// UpdateResult is the response of a Update operations. Call its Extract method
// to interpret it as a Cluster.
type UpdateResult struct {
	commonResult
}

// GetPolicyResult is the response of a Get operations. Call its Extract method
// to interpret it as a ClusterPolicy.
type GetPolicyResult struct {
	gophercloud.Result
}

// Extract interprets a GetPolicyResult as a ClusterPolicy.
func (r GetPolicyResult) Extract() (*ClusterPolicy, error) {
	var s struct {
		ClusterPolicy *ClusterPolicy `json:"cluster_policy"`
	}
	err := r.ExtractInto(&s)
	return s.ClusterPolicy, err
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// ClusterPage contains a single page of all clusters from a List call.
type ClusterPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Clusters contains any results.
func (page ClusterPage) IsEmpty() (bool, error) {
	clusters, err := ExtractClusters(page)
	return len(clusters) == 0, err
}

// ClusterPolicyPage contains a single page of all policies from a List call
type ClusterPolicyPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of ClusterPolicies contains any
// results.
func (page ClusterPolicyPage) IsEmpty() (bool, error) {
	clusterPolicies, err := ExtractClusterPolicies(page)
	return len(clusterPolicies) == 0, err
}

// ActionResult is the response of Senlin actions. Call its Extract method to
// obtain the Action ID of the action.
type ActionResult struct {
	gophercloud.Result
}

// Extract interprets any Action result as an Action.
func (r ActionResult) Extract() (string, error) {
	var s struct {
		Action string `json:"action"`
	}
	err := r.ExtractInto(&s)
	return s.Action, err
}

type CollectResult struct {
	gophercloud.Result
}

// ExtractClusters returns a slice of Clusters from the List operation.
func ExtractClusters(r pagination.Page) ([]Cluster, error) {
	var s struct {
		Clusters []Cluster `json:"clusters"`
	}
	err := (r.(ClusterPage)).ExtractInto(&s)
	return s.Clusters, err
}

// ExtractClusterPolicies returns a slice of ClusterPolicies from the
// ListClusterPolicies operation.
func ExtractClusterPolicies(r pagination.Page) ([]ClusterPolicy, error) {
	var s struct {
		ClusterPolicies []ClusterPolicy `json:"cluster_policies"`
	}
	err := (r.(ClusterPolicyPage)).ExtractInto(&s)
	return s.ClusterPolicies, err
}

// Extract returns collected attributes across a cluster
func (r CollectResult) Extract() ([]ClusterAttributes, error) {
	var s struct {
		Attributes []ClusterAttributes `json:"cluster_attributes"`
	}
	err := r.ExtractInto(&s)
	return s.Attributes, err
}
