//
// Copyright 2021, Matej Velikonja
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

// ProjectClustersService handles communication with the
// project clusters related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html
type ProjectClustersService struct {
	client *Client
}

// ProjectCluster represents a GitLab Project Cluster.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/project_clusters.html
type ProjectCluster struct {
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
	Project            *Project            `json:"project"`
}

func (v ProjectCluster) String() string {
	return Stringify(v)
}

// PlatformKubernetes represents a GitLab Project Cluster PlatformKubernetes.
type PlatformKubernetes struct {
	APIURL            string `json:"api_url"`
	Token             string `json:"token"`
	CaCert            string `json:"ca_cert"`
	Namespace         string `json:"namespace"`
	AuthorizationType string `json:"authorization_type"`
}

// ManagementProject represents a GitLab Project Cluster management_project.
type ManagementProject struct {
	ID                int        `json:"id"`
	Description       string     `json:"description"`
	Name              string     `json:"name"`
	NameWithNamespace string     `json:"name_with_namespace"`
	Path              string     `json:"path"`
	PathWithNamespace string     `json:"path_with_namespace"`
	CreatedAt         *time.Time `json:"created_at"`
}

// ListClusters gets a list of all clusters in a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#list-project-clusters
func (s *ProjectClustersService) ListClusters(pid interface{}, options ...RequestOptionFunc) ([]*ProjectCluster, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/clusters", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var pcs []*ProjectCluster
	resp, err := s.client.Do(req, &pcs)
	if err != nil {
		return nil, resp, err
	}

	return pcs, resp, err
}

// GetCluster gets a cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#get-a-single-project-cluster
func (s *ProjectClustersService) GetCluster(pid interface{}, cluster int, options ...RequestOptionFunc) (*ProjectCluster, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/clusters/%d", pathEscape(project), cluster)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pc := new(ProjectCluster)
	resp, err := s.client.Do(req, &pc)
	if err != nil {
		return nil, resp, err
	}

	return pc, resp, err
}

// AddClusterOptions represents the available AddCluster() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#add-existing-cluster-to-project
type AddClusterOptions struct {
	Name                *string                       `url:"name,omitempty" json:"name,omitempty"`
	Domain              *string                       `url:"domain,omitempty" json:"domain,omitempty"`
	Enabled             *bool                         `url:"enabled,omitempty" json:"enabled,omitempty"`
	Managed             *bool                         `url:"managed,omitempty" json:"managed,omitempty"`
	EnvironmentScope    *string                       `url:"environment_scope,omitempty" json:"environment_scope,omitempty"`
	PlatformKubernetes  *AddPlatformKubernetesOptions `url:"platform_kubernetes_attributes,omitempty" json:"platform_kubernetes_attributes,omitempty"`
	ManagementProjectID *string                       `url:"management_project_id,omitempty" json:"management_project_id,omitempty"`
}

// AddPlatformKubernetesOptions represents the available PlatformKubernetes options for adding.
type AddPlatformKubernetesOptions struct {
	APIURL            *string `url:"api_url,omitempty" json:"api_url,omitempty"`
	Token             *string `url:"token,omitempty" json:"token,omitempty"`
	CaCert            *string `url:"ca_cert,omitempty" json:"ca_cert,omitempty"`
	Namespace         *string `url:"namespace,omitempty" json:"namespace,omitempty"`
	AuthorizationType *string `url:"authorization_type,omitempty" json:"authorization_type,omitempty"`
}

// AddCluster adds an existing cluster to the project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#add-existing-cluster-to-project
func (s *ProjectClustersService) AddCluster(pid interface{}, opt *AddClusterOptions, options ...RequestOptionFunc) (*ProjectCluster, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/clusters/user", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pc := new(ProjectCluster)
	resp, err := s.client.Do(req, pc)
	if err != nil {
		return nil, resp, err
	}

	return pc, resp, err
}

// EditClusterOptions represents the available EditCluster() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#edit-project-cluster
type EditClusterOptions struct {
	Name                *string                        `url:"name,omitempty" json:"name,omitempty"`
	Domain              *string                        `url:"domain,omitempty" json:"domain,omitempty"`
	EnvironmentScope    *string                        `url:"environment_scope,omitempty" json:"environment_scope,omitempty"`
	ManagementProjectID *string                        `url:"management_project_id,omitempty" json:"management_project_id,omitempty"`
	PlatformKubernetes  *EditPlatformKubernetesOptions `url:"platform_kubernetes_attributes,omitempty" json:"platform_kubernetes_attributes,omitempty"`
}

// EditPlatformKubernetesOptions represents the available PlatformKubernetes options for editing.
type EditPlatformKubernetesOptions struct {
	APIURL    *string `url:"api_url,omitempty" json:"api_url,omitempty"`
	Token     *string `url:"token,omitempty" json:"token,omitempty"`
	CaCert    *string `url:"ca_cert,omitempty" json:"ca_cert,omitempty"`
	Namespace *string `url:"namespace,omitempty" json:"namespace,omitempty"`
}

// EditCluster updates an existing project cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#edit-project-cluster
func (s *ProjectClustersService) EditCluster(pid interface{}, cluster int, opt *EditClusterOptions, options ...RequestOptionFunc) (*ProjectCluster, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/clusters/%d", pathEscape(project), cluster)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pc := new(ProjectCluster)
	resp, err := s.client.Do(req, pc)
	if err != nil {
		return nil, resp, err
	}

	return pc, resp, err
}

// DeleteCluster deletes an existing project cluster.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_clusters.html#delete-project-cluster
func (s *ProjectClustersService) DeleteCluster(pid interface{}, cluster int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/clusters/%d", pathEscape(project), cluster)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
