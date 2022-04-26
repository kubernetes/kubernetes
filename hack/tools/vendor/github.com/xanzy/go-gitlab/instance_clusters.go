//
// Copyright 2021, Serena Fang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"fmt"
	"time"
)

// InstanceClustersService handles communication with the
// instance clusters related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html
type InstanceClustersService struct {
	client *Client
}

// InstanceCluster represents a GitLab Instance Cluster.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/instance_clusters.html
type InstanceCluster struct {
	ID                 int                 `json:"id"`
	Name               string              `json:"name"`
	Domain             string              `json:"domain"`
	Managed            bool                `json:"managed"`
	CreatedAt          *time.Time          `json:"created_at"`
	ProviderType       string              `json:"provider_type"`
	PlatformType       string              `json:"platform_type"`
	EnvironmentScope   string              `json:"environment_scope"`
	ClusterType        string              `json:"cluster_type"`
	User               *User               `json:"user"`
	PlatformKubernetes *PlatformKubernetes `json:"platform_kubernetes"`
	ManagementProject  *ManagementProject  `json:"management_project"`
}

func (v InstanceCluster) String() string {
	return Stringify(v)
}

// ListClusters gets a list of all instance clusters.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html#list-instance-clusters
func (s *InstanceClustersService) ListClusters(options ...RequestOptionFunc) ([]*InstanceCluster, *Response, error) {
	u := "admin/clusters"

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var ics []*InstanceCluster
	resp, err := s.client.Do(req, &ics)
	if err != nil {
		return nil, resp, err
	}

	return ics, resp, err
}

// GetCluster gets an instance cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html#get-a-single-instance-cluster
func (s *InstanceClustersService) GetCluster(cluster int, options ...RequestOptionFunc) (*InstanceCluster, *Response, error) {
	u := fmt.Sprintf("admin/clusters/%d", cluster)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ic := new(InstanceCluster)
	resp, err := s.client.Do(req, &ic)
	if err != nil {
		return nil, resp, err
	}

	return ic, resp, err
}

// AddCluster adds an existing cluster to the instance.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html#add-existing-instance-cluster
func (s *InstanceClustersService) AddCluster(opt *AddClusterOptions, options ...RequestOptionFunc) (*InstanceCluster, *Response, error) {
	u := "admin/clusters/add"

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ic := new(InstanceCluster)
	resp, err := s.client.Do(req, ic)
	if err != nil {
		return nil, resp, err
	}

	return ic, resp, err
}

// EditCluster updates an existing instance cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html#edit-instance-cluster
func (s *InstanceClustersService) EditCluster(cluster int, opt *EditClusterOptions, options ...RequestOptionFunc) (*InstanceCluster, *Response, error) {
	u := fmt.Sprintf("admin/clusters/%d", cluster)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ic := new(InstanceCluster)
	resp, err := s.client.Do(req, ic)
	if err != nil {
		return nil, resp, err
	}

	return ic, resp, err
}

// DeleteCluster deletes an existing instance cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_clusters.html#delete-instance-cluster
func (s *InstanceClustersService) DeleteCluster(cluster int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("admin/clusters/%d", cluster)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
