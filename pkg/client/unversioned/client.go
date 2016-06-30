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

package unversioned

import (
	"net"
	"net/url"
	"strings"

	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
)

// Interface holds the methods for clients of Kubernetes,
// an interface to allow mock testing.
type Interface interface {
	PodsNamespacer
	PodTemplatesNamespacer
	ReplicationControllersNamespacer
	ServicesNamespacer
	EndpointsNamespacer
	NodesInterface
	EventNamespacer
	LimitRangesNamespacer
	ResourceQuotasNamespacer
	ServiceAccountsNamespacer
	SecretsNamespacer
	NamespacesInterface
	PersistentVolumesInterface
	PersistentVolumeClaimsNamespacer
	ComponentStatusesInterface
	ConfigMapsNamespacer
	Autoscaling() AutoscalingInterface
	Batch() BatchInterface
	Extensions() ExtensionsInterface
	Rbac() RbacInterface
	Discovery() discovery.DiscoveryInterface
	Certificates() CertificatesInterface
}

func (c *Client) ReplicationControllers(namespace string) ReplicationControllerInterface {
	return newReplicationControllers(c, namespace)
}

func (c *Client) Nodes() NodeInterface {
	return newNodes(c)
}

func (c *Client) Events(namespace string) EventInterface {
	return newEvents(c, namespace)
}

func (c *Client) Endpoints(namespace string) EndpointsInterface {
	return newEndpoints(c, namespace)
}

func (c *Client) Pods(namespace string) PodInterface {
	return newPods(c, namespace)
}

func (c *Client) PodTemplates(namespace string) PodTemplateInterface {
	return newPodTemplates(c, namespace)
}

func (c *Client) Services(namespace string) ServiceInterface {
	return newServices(c, namespace)
}
func (c *Client) LimitRanges(namespace string) LimitRangeInterface {
	return newLimitRanges(c, namespace)
}

func (c *Client) ResourceQuotas(namespace string) ResourceQuotaInterface {
	return newResourceQuotas(c, namespace)
}

func (c *Client) ServiceAccounts(namespace string) ServiceAccountsInterface {
	return newServiceAccounts(c, namespace)
}

func (c *Client) Secrets(namespace string) SecretsInterface {
	return newSecrets(c, namespace)
}

func (c *Client) Namespaces() NamespaceInterface {
	return newNamespaces(c)
}

func (c *Client) PersistentVolumes() PersistentVolumeInterface {
	return newPersistentVolumes(c)
}

func (c *Client) PersistentVolumeClaims(namespace string) PersistentVolumeClaimInterface {
	return newPersistentVolumeClaims(c, namespace)
}

func (c *Client) ComponentStatuses() ComponentStatusInterface {
	return newComponentStatuses(c)
}

func (c *Client) ConfigMaps(namespace string) ConfigMapsInterface {
	return newConfigMaps(c, namespace)
}

// Client is the implementation of a Kubernetes client.
type Client struct {
	*restclient.RESTClient
	*AutoscalingClient
	*BatchClient
	*ExtensionsClient
	*AppsClient
	*PolicyClient
	*RbacClient
	*discovery.DiscoveryClient
	*CertificatesClient
}

// IsTimeout tests if this is a timeout error in the underlying transport.
// This is unbelievably ugly.
// See: http://stackoverflow.com/questions/23494950/specifically-check-for-timeout-error for details
func IsTimeout(err error) bool {
	if err == nil {
		return false
	}
	switch err := err.(type) {
	case *url.Error:
		if err, ok := err.Err.(net.Error); ok {
			return err.Timeout()
		}
	case net.Error:
		return err.Timeout()
	}

	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}
	return false
}

func (c *Client) Autoscaling() AutoscalingInterface {
	return c.AutoscalingClient
}

func (c *Client) Batch() BatchInterface {
	return c.BatchClient
}

func (c *Client) Extensions() ExtensionsInterface {
	return c.ExtensionsClient
}

func (c *Client) Apps() AppsInterface {
	return c.AppsClient
}

func (c *Client) Rbac() RbacInterface {
	return c.RbacClient
}

func (c *Client) Discovery() discovery.DiscoveryInterface {
	return c.DiscoveryClient
}

func (c *Client) Certificates() CertificatesInterface {
	return c.CertificatesClient
}
