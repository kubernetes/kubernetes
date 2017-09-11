/*
Copyright 2017 The Kubernetes Authors.

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

package networkpolicy

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/networking"
)

// Registry is an interface for things that know how to store NetworkPolicies.
type Registry interface {
	ListNetworkPolicies(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*networking.NetworkPolicyList, error)
	CreateNetworkPolicy(ctx genericapirequest.Context, np *networking.NetworkPolicy) error
	UpdateNetworkPolicy(ctx genericapirequest.Context, np *networking.NetworkPolicy) error
	GetNetworkPolicy(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*networking.NetworkPolicy, error)
	DeleteNetworkPolicy(ctx genericapirequest.Context, name string) error
	WatchNetworkPolicies(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
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

func (s *storage) ListNetworkPolicies(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*networking.NetworkPolicyList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*networking.NetworkPolicyList), nil
}

func (s *storage) CreateNetworkPolicy(ctx genericapirequest.Context, np *networking.NetworkPolicy) error {
	_, err := s.Create(ctx, np, false)
	return err
}

func (s *storage) UpdateNetworkPolicy(ctx genericapirequest.Context, np *networking.NetworkPolicy) error {
	_, _, err := s.Update(ctx, np.Name, rest.DefaultUpdatedObjectInfo(np, api.Scheme))
	return err
}

func (s *storage) WatchNetworkPolicies(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetNetworkPolicy(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*networking.NetworkPolicy, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*networking.NetworkPolicy), nil
}

func (s *storage) DeleteNetworkPolicy(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
