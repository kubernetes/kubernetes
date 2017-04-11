/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package deployment

import (
	"fmt"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// Registry is an interface for things that know how to store Deployments.
type Registry interface {
	ListDeployments(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*extensions.DeploymentList, error)
	GetDeployment(ctx genericapirequest.Context, deploymentID string, options *metav1.GetOptions) (*extensions.Deployment, error)
	CreateDeployment(ctx genericapirequest.Context, deployment *extensions.Deployment) (*extensions.Deployment, error)
	UpdateDeployment(ctx genericapirequest.Context, deployment *extensions.Deployment) (*extensions.Deployment, error)
	DeleteDeployment(ctx genericapirequest.Context, deploymentID string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListDeployments(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*extensions.DeploymentList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.DeploymentList), err
}

func (s *storage) GetDeployment(ctx genericapirequest.Context, deploymentID string, options *metav1.GetOptions) (*extensions.Deployment, error) {
	obj, err := s.Get(ctx, deploymentID, options)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), nil
}

func (s *storage) CreateDeployment(ctx genericapirequest.Context, deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, err := s.Create(ctx, deployment)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), nil
}

func (s *storage) UpdateDeployment(ctx genericapirequest.Context, deployment *extensions.Deployment) (*extensions.Deployment, error) {
	obj, _, err := s.Update(ctx, deployment.Name, rest.DefaultUpdatedObjectInfo(deployment, api.Scheme))
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Deployment), nil
}

func (s *storage) DeleteDeployment(ctx genericapirequest.Context, deploymentID string) error {
	_, _, err := s.Delete(ctx, deploymentID, nil)
	return err
}
