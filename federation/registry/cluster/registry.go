/*
Copyright 2016 The Kubernetes Authors.

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

package cluster

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
)

// Registry is an interface implemented by things that know how to store Cluster objects.
type Registry interface {
	ListClusters(ctx genericapirequest.Context, options *api.ListOptions) (*federation.ClusterList, error)
	WatchCluster(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error)
	GetCluster(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*federation.Cluster, error)
	CreateCluster(ctx genericapirequest.Context, cluster *federation.Cluster) error
	UpdateCluster(ctx genericapirequest.Context, cluster *federation.Cluster) error
	DeleteCluster(ctx genericapirequest.Context, name string) error
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

func (s *storage) ListClusters(ctx genericapirequest.Context, options *api.ListOptions) (*federation.ClusterList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.ClusterList), nil
}

func (s *storage) WatchCluster(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetCluster(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*federation.Cluster, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.Cluster), nil
}

func (s *storage) CreateCluster(ctx genericapirequest.Context, cluster *federation.Cluster) error {
	_, err := s.Create(ctx, cluster)
	return err
}

func (s *storage) UpdateCluster(ctx genericapirequest.Context, cluster *federation.Cluster) error {
	_, _, err := s.Update(ctx, cluster.Name, rest.DefaultUpdatedObjectInfo(cluster, api.Scheme))
	return err
}

func (s *storage) DeleteCluster(ctx genericapirequest.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
