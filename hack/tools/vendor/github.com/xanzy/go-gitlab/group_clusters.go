//
// Copyright 2021, Paul Shoemaker
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

// GroupClustersService handles communication with the
// group clusters related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html
type GroupClustersService struct {
	client *Client
}

// GroupCluster represents a GitLab Group Cluster.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/group_clusters.html
type GroupCluster struct {
	ID                 int                 `json:"id"`
	Name               string              `json:"name"`
	Domain             string              `json:"domain"`
	CreatedAt          *time.Time          `json:"created_at"`
	ProviderType       string              `json:"provider_type"`
	PlatformType       string              `json:"platform_type"`
	EnvironmentScope   string              `json:"environment_scope"`
	ClusterType        string              `json:"cluster_type"`
	User               *User               `json:"user"`
	PlatformKubernetes *PlatformKubernetes `json:"platform_kubernetes"`
	ManagementProject  *ManagementProject  `json:"management_project"`
	Group              *Group              `json:"group"`
}

func (v GroupCluster) String() string {
	return Stringify(v)
}

// ListClusters gets a list of all clusters in a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#list-group-clusters
func (s *GroupClustersService) ListClusters(pid interface{}, options ...RequestOptionFunc) ([]*GroupCluster, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/clusters", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var pcs []*GroupCluster
	resp, err := s.client.Do(req, &pcs)
	if err != nil {
		return nil, resp, err
	}

	return pcs, resp, err
}

// GetCluster gets a cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#get-a-single-group-cluster
func (s *GroupClustersService) GetCluster(pid interface{}, cluster int, options ...RequestOptionFunc) (*GroupCluster, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/clusters/%d", pathEscape(group), cluster)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	gc := new(GroupCluster)
	resp, err := s.client.Do(req, &gc)
	if err != nil {
		return nil, resp, err
	}

	return gc, resp, err
}

// AddGroupClusterOptions represents the available AddCluster() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#add-existing-cluster-to-group
type AddGroupClusterOptions struct {
	Name                *string                            `url:"name,omitempty" json:"name,omitempty"`
	Domain              *string                            `url:"domain,omitempty" json:"domain,omitempty"`
	ManagementProjectID *string                            `url:"management_project_id,omitempty" json:"management_project_id,omitempty"`
	Enabled             *bool                              `url:"enabled,omitempty" json:"enabled,omitempty"`
	Managed             *bool                              `url:"managed,omitempty" json:"managed,omitempty"`
	EnvironmentScope    *string                            `url:"environment_scope,omitempty" json:"environment_scope,omitempty"`
	PlatformKubernetes  *AddGroupPlatformKubernetesOptions `url:"platform_kubernetes_attributes,omitempty" json:"platform_kubernetes_attributes,omitempty"`
}

// AddGroupPlatformKubernetesOptions represents the available PlatformKubernetes options for adding.
type AddGroupPlatformKubernetesOptions struct {
	APIURL            *string `url:"api_url,omitempty" json:"api_url,omitempty"`
	Token             *string `url:"token,omitempty" json:"token,omitempty"`
	CaCert            *string `url:"ca_cert,omitempty" json:"ca_cert,omitempty"`
	Namespace         *string `url:"namespace,omitempty" json:"namespace,omitempty"`
	AuthorizationType *string `url:"authorization_type,omitempty" json:"authorization_type,omitempty"`
}

// AddCluster adds an existing cluster to the group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#add-existing-cluster-to-group
func (s *GroupClustersService) AddCluster(pid interface{}, opt *AddGroupClusterOptions, options ...RequestOptionFunc) (*GroupCluster, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/clusters/user", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gc := new(GroupCluster)
	resp, err := s.client.Do(req, gc)
	if err != nil {
		return nil, resp, err
	}

	return gc, resp, err
}

// EditGroupClusterOptions represents the available EditCluster() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#edit-group-cluster
type EditGroupClusterOptions struct {
	Name                *string                             `url:"name,omitempty" json:"name,omitempty"`
	Domain              *string                             `url:"domain,omitempty" json:"domain,omitempty"`
	EnvironmentScope    *string                             `url:"environment_scope,omitempty" json:"environment_scope,omitempty"`
	PlatformKubernetes  *EditGroupPlatformKubernetesOptions `url:"platform_kubernetes_attributes,omitempty" json:"platform_kubernetes_attributes,omitempty"`
	ManagementProjectID *string                             `url:"management_project_id,omitempty" json:"management_project_id,omitempty"`
}

// EditGroupPlatformKubernetesOptions represents the available PlatformKubernetes options for editing.
type EditGroupPlatformKubernetesOptions struct {
	APIURL *string `url:"api_url,omitempty" json:"api_url,omitempty"`
	Token  *string `url:"token,omitempty" json:"token,omitempty"`
	CaCert *string `url:"ca_cert,omitempty" json:"ca_cert,omitempty"`
}

// EditCluster updates an existing group cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#edit-group-cluster
func (s *GroupClustersService) EditCluster(pid interface{}, cluster int, opt *EditGroupClusterOptions, options ...RequestOptionFunc) (*GroupCluster, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/clusters/%d", pathEscape(group), cluster)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gc := new(GroupCluster)
	resp, err := s.client.Do(req, gc)
	if err != nil {
		return nil, resp, err
	}

	return gc, resp, err
}

// DeleteCluster deletes an existing group cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_clusters.html#delete-group-cluster
func (s *GroupClustersService) DeleteCluster(pid interface{}, cluster int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/clusters/%d", pathEscape(group), cluster)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
