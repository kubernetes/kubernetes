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

package securitycontextconstraints

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store SecurityContextConstraints objects.
type Registry interface {
	// ListSecurityContextConstraints obtains a list of SecurityContextConstraints having labels which match selector.
	ListSecurityContextConstraints(ctx api.Context, options *api.ListOptions) (*api.SecurityContextConstraintsList, error)
	// Watch for new/changed/deleted SecurityContextConstraints
	WatchSecurityContextConstraints(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
	// Get a specific SecurityContextConstraints
	GetSecurityContextConstraint(ctx api.Context, name string) (*api.SecurityContextConstraints, error)
	// Create a SecurityContextConstraints based on a specification.
	CreateSecurityContextConstraint(ctx api.Context, scc *api.SecurityContextConstraints) error
	// Update an existing SecurityContextConstraints
	UpdateSecurityContextConstraint(ctx api.Context, scc *api.SecurityContextConstraints) error
	// Delete an existing SecurityContextConstraints
	DeleteSecurityContextConstraint(ctx api.Context, name string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListSecurityContextConstraints(ctx api.Context, options *api.ListOptions) (*api.SecurityContextConstraintsList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraintsList), nil
}

func (s *storage) WatchSecurityContextConstraints(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetSecurityContextConstraint(ctx api.Context, name string) (*api.SecurityContextConstraints, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), nil
}

func (s *storage) CreateSecurityContextConstraint(ctx api.Context, scc *api.SecurityContextConstraints) error {
	_, err := s.Create(ctx, scc)
	return err
}

func (s *storage) UpdateSecurityContextConstraint(ctx api.Context, scc *api.SecurityContextConstraints) error {
	_, _, err := s.Update(ctx, scc.Name, rest.DefaultUpdatedObjectInfo(scc, api.Scheme))
	return err
}

func (s *storage) DeleteSecurityContextConstraint(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
