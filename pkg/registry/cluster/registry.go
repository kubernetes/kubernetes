package cluster

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store Cluster objects.
type Registry interface {
	ListClusters(ctx api.Context, options *api.ListOptions) (*api.ClusterList, error)
	WatchCluster(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
	GetCluster(ctx api.Context, clusterID string) (*api.Cluster, error)
	CreateCluster(ctx api.Context, cluster *api.Cluster) error
	UpdateCluster(ctx api.Context, cluster *api.Cluster) error
	DeleteCluster(ctx api.Context, clusterID string) error
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

func (s *storage) ListClusters(ctx api.Context, options *api.ListOptions) (*api.ClusterList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ClusterList), nil
}

func (s *storage) WatchCluster(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetCluster(ctx api.Context, clusterID string) (*api.Cluster, error) {
	obj, err := s.Get(ctx, clusterID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Cluster), nil
}

func (s *storage) CreateCluster(ctx api.Context, cluster *api.Cluster) error {
	_, err := s.Create(ctx, cluster)
	return err
}

func (s *storage) UpdateCluster(ctx api.Context, cluster *api.Cluster) error {
	_, _, err := s.Update(ctx, cluster)
	return err
}

func (s *storage) DeleteCluster(ctx api.Context, clusterID string) error {
	_, err := s.Delete(ctx, clusterID, nil)
	return err
}
