package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

type ClustersInterface interface {
	Clusters() ClusterInterface
}

type ClusterInterface interface {
	Get(name string) (*api.Cluster, error)
	Create (cluster *api.Cluster) (*api.Cluster, error)
	List(opts api.ListOptions) (*api.ClusterList, error)
	Delete(name string) error
	Update(*api.Cluster) (*api.Cluster, error)
	UpdateStatus(*api.Cluster) (*api.Cluster, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

type clusters struct {
	r *Client
}

func newClusters(c *Client) *clusters {
	return &clusters{c}
}

func (c *clusters) resourceName() string {
	return "clusters"
}

func (c *clusters) Get(name string) (*api.Cluster, error) {
	result := &api.Cluster{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

func (c *clusters) Create(cluster *api.Cluster) (*api.Cluster, error) {
	result := &api.Cluster{}
	err := c.r.Post().Resource(c.resourceName()).Body(cluster).Do().Into(result)
	return result, err
}

func (c *clusters) List(opts api.ListOptions) (*api.ClusterList, error) {
	result := &api.ClusterList{}
	err := c.r.Get().Resource(c.resourceName()).VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return result, err
}


func (c *clusters) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

func (c *clusters) Update(cluster *api.Cluster) (*api.Cluster, error) {
	result := &api.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).Body(cluster).Do().Into(result)
	return result, err
}

func (c *clusters) UpdateStatus(cluster *api.Cluster) (*api.Cluster, error) {
	result := &api.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).SubResource("status").Body(cluster).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested cluster.
func (c *clusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
	Prefix("watch").
	Namespace(api.NamespaceAll).
	Resource(c.resourceName()).
	VersionedParams(&opts, api.ParameterCodec).
	Watch()
}
