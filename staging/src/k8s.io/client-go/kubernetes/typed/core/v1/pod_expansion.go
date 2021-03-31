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

package v1

import (
	"context"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
)

// The PodExpansion interface allows manually adding extra methods to the PodInterface.
type PodExpansion interface {
	Bind(ctx context.Context, binding *v1.Binding, opts metav1.CreateOptions) error
	// Evict submits a policy/v1beta1 Eviction request to the pod's eviction subresource.
	// Equivalent to calling EvictV1beta1.
	// Deprecated: Use EvictV1() (supported in 1.22+) or EvictV1beta1().
	Evict(ctx context.Context, eviction *policyv1beta1.Eviction) error
	// EvictV1 submits a policy/v1 Eviction request to the pod's eviction subresource.
	// Supported in 1.22+.
	EvictV1(ctx context.Context, eviction *policyv1.Eviction) error
	// EvictV1beta1 submits a policy/v1beta1 Eviction request to the pod's eviction subresource.
	// Supported in 1.22+.
	EvictV1beta1(ctx context.Context, eviction *policyv1beta1.Eviction) error
	GetLogs(name string, opts *v1.PodLogOptions) *restclient.Request
	ProxyGet(scheme, name, port, path string, params map[string]string) restclient.ResponseWrapper
}

// Bind applies the provided binding to the named pod in the current namespace (binding.Namespace is ignored).
func (c *pods) Bind(ctx context.Context, binding *v1.Binding, opts metav1.CreateOptions) error {
	return c.client.Post().Namespace(c.ns).Resource("pods").Name(binding.Name).VersionedParams(&opts, scheme.ParameterCodec).SubResource("binding").Body(binding).Do(ctx).Error()
}

// Evict submits a policy/v1beta1 Eviction request to the pod's eviction subresource.
// Equivalent to calling EvictV1beta1.
// Deprecated: Use EvictV1() (supported in 1.22+) or EvictV1beta1().
func (c *pods) Evict(ctx context.Context, eviction *policyv1beta1.Eviction) error {
	return c.client.Post().Namespace(c.ns).Resource("pods").Name(eviction.Name).SubResource("eviction").Body(eviction).Do(ctx).Error()
}

func (c *pods) EvictV1beta1(ctx context.Context, eviction *policyv1beta1.Eviction) error {
	return c.client.Post().Namespace(c.ns).Resource("pods").Name(eviction.Name).SubResource("eviction").Body(eviction).Do(ctx).Error()
}

func (c *pods) EvictV1(ctx context.Context, eviction *policyv1.Eviction) error {
	return c.client.Post().Namespace(c.ns).Resource("pods").Name(eviction.Name).SubResource("eviction").Body(eviction).Do(ctx).Error()
}

// Get constructs a request for getting the logs for a pod
func (c *pods) GetLogs(name string, opts *v1.PodLogOptions) *restclient.Request {
	return c.client.Get().Namespace(c.ns).Name(name).Resource("pods").SubResource("log").VersionedParams(opts, scheme.ParameterCodec)
}

// ProxyGet returns a response of the pod by calling it through the proxy.
func (c *pods) ProxyGet(scheme, name, port, path string, params map[string]string) restclient.ResponseWrapper {
	request := c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		SubResource("proxy").
		Name(net.JoinSchemeNamePort(scheme, name, port)).
		Suffix(path)
	for k, v := range params {
		request = request.Param(k, v)
	}
	return request
}
