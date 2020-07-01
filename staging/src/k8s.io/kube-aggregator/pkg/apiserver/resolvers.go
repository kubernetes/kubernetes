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

package apiserver

import (
	"fmt"
	"net/url"

	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/util/proxy"
	listersv1 "k8s.io/client-go/listers/core/v1"
)

// A ServiceResolver knows how to get a URL given a service.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string, port int32) (*url.URL, error)
}

// RetryServiceResolver extends ServiceResolver interface by providing a few additional methods for supporting requests retries
type RetryServiceResolver interface {
	ServiceResolver

	// ResolveEndpointWithVisited resolves an endpoint excluding already visited ones. Facilitates supporting retry mechanisms.
	ResolveEndpointWithVisited(namespace, name string, port int32, visitedEPs []*url.URL) (*url.URL, error)

	// EndpointCount return the total number of endpoints for the given service
	EndpointCount(namespace, name string, port int32) (int, error)
}

// NewEndpointServiceResolver returns a ServiceResolver that chooses one of the
// service's endpoints.
func NewEndpointServiceResolver(services listersv1.ServiceLister, endpoints listersv1.EndpointsLister) ServiceResolver {
	return &aggregatorEndpointRouting{
		services:  services,
		endpoints: endpoints,
	}
}

type aggregatorEndpointRouting struct {
	services  listersv1.ServiceLister
	endpoints listersv1.EndpointsLister
}

func (r *aggregatorEndpointRouting) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return proxy.ResolveEndpoint(r.services, r.endpoints, namespace, name, port)
}

// NewClusterIPServiceResolver returns a ServiceResolver that directly calls the
// service's cluster IP.
func NewClusterIPServiceResolver(services listersv1.ServiceLister) ServiceResolver {
	return &aggregatorClusterRouting{
		services: services,
	}
}

type aggregatorClusterRouting struct {
	services listersv1.ServiceLister
}

func (r *aggregatorClusterRouting) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return proxy.ResolveCluster(r.services, namespace, name, port)
}

// NewLoopbackServiceResolver returns a ServiceResolver that routes
// the kubernetes/default service with port 443 to loopback.
func NewLoopbackServiceResolver(delegate ServiceResolver, host *url.URL) ServiceResolver {
	return &loopbackResolver{
		delegate: delegate,
		host:     host,
	}
}

type loopbackResolver struct {
	delegate ServiceResolver
	host     *url.URL
}

func (r *loopbackResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	if namespace == "default" && name == "kubernetes" && port == 443 {
		return r.host, nil
	}
	if r.delegate == nil {
		return nil, nil
	}
	return r.delegate.ResolveEndpoint(namespace, name, port)
}

type fakeLoopbackResolver struct{}

func (f *fakeLoopbackResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return nil, nil
}

// NewExtendedEndpointServiceResolver returns a service resolver that extends ServiceResolver interface by providing a few additional methods for supporting requests retries
func NewExtendedEndpointServiceResolver(services listersv1.ServiceLister, endpoints listersv1.EndpointsLister, loopbackResolver ServiceResolver) ServiceResolver {
	if loopbackResolver == nil {
		loopbackResolver = &fakeLoopbackResolver{}
	}
	return &serviceResolver{
		services:         services,
		endpoints:        endpoints,
		loopbackResolver: loopbackResolver,
	}
}

type serviceResolver struct {
	services         listersv1.ServiceLister
	endpoints        listersv1.EndpointsLister
	loopbackResolver ServiceResolver
}

// ResolveEndpoint resolves (randomly) an endpoint to a given service.
//
// Note: Kube uses one service resolver for webhooks and the aggregator this method satisfies webhook.ServiceResolver interface
func (r *serviceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	localEndpoint, err := r.loopbackResolver.ResolveEndpoint(namespace, name, port)
	if err != nil {
		return nil, err
	}
	if localEndpoint != nil {
		return localEndpoint, nil
	}

	return proxy.ResolveEndpoint(r.services, r.endpoints, namespace, name, port)
}

// ResolveEndpointWithVisited resolves an endpoint excluding already visited ones. Facilitates supporting retry mechanisms.
func (r *serviceResolver) ResolveEndpointWithVisited(namespace, name string, port int32, visitedEPs []*url.URL) (*url.URL, error) {
	localEndpoint, err := r.loopbackResolver.ResolveEndpoint(namespace, name, port)
	if err == nil && localEndpoint != nil {
		return localEndpoint, nil
	}

	return proxy.ResolveEndpoint(r.services, r.endpoints, namespace, name, port, visitedEPs...)
}

// EndpointCount return the total number of endpoints for the given service
func (r *serviceResolver) EndpointCount(namespace, name string, port int32) (int, error) {
	localEndpoint, err := r.loopbackResolver.ResolveEndpoint(namespace, name, port)
	if err == nil && localEndpoint != nil {
		return 1, nil
	}

	allEndpoints, err := proxy.ResolveEndpoints(r.services, r.endpoints, namespace, name, port)
	if err != nil {
		return 0, err
	}

	if len(allEndpoints) == 0 {
		return 0, kerrors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q/%q", namespace, name))
	}

	return len(allEndpoints), nil
}
