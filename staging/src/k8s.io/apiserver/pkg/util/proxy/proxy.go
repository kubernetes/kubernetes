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

package proxy

import (
	"fmt"
	"math/rand"
	"net"
	"net/url"
	"strconv"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	listersv1 "k8s.io/client-go/listers/core/v1"
)

// findServicePort finds the service port by name or numerically.
func findServicePort(svc *v1.Service, port int32) (*v1.ServicePort, error) {
	for _, svcPort := range svc.Spec.Ports {
		if svcPort.Port == port {
			return &svcPort, nil
		}
	}
	return nil, errors.NewServiceUnavailable(fmt.Sprintf("no service port %d found for service %q", port, svc.Name))
}

// ResolveEndpoint returns a URL to which one can send traffic for the specified service.
// This method accepts an optional parameter (excludedEndpoints) for specifying a list of endpoints that must be excluded.
func ResolveEndpoint(services listersv1.ServiceLister, endpoints listersv1.EndpointsLister, namespace, id string, port int32, excludedEndpoints ...*url.URL) (*url.URL, error) {
	allEndpoints, err := ResolveEndpoints(services, endpoints, namespace, id, port, excludedEndpoints...)
	if err != nil {
		return nil, err
	}

	if len(allEndpoints) == 0 {
		return nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
	}

	return allEndpoints[rand.Intn(len(allEndpoints))], nil
}

// ResolveEndpoint returns all URLs to which one can send traffic for the specified service.
// This method accepts an optional parameter (excludedEndpoints) for specifying a list of endpoints that must be excluded.
func ResolveEndpoints(services listersv1.ServiceLister, endpoints listersv1.EndpointsLister, namespace, id string, port int32, excludedEndpoints ...*url.URL) ([]*url.URL, error) {
	svc, err := services.Services(namespace).Get(id)
	if err != nil {
		return nil, err
	}

	svcPort, err := findServicePort(svc, port)
	if err != nil {
		return nil, err
	}

	switch {
	case svc.Spec.Type == v1.ServiceTypeClusterIP, svc.Spec.Type == v1.ServiceTypeLoadBalancer, svc.Spec.Type == v1.ServiceTypeNodePort:
		// these are fine
	default:
		return nil, fmt.Errorf("unsupported service type %q", svc.Spec.Type)
	}

	eps, err := endpoints.Endpoints(namespace).Get(svc.Name)
	if err != nil {
		return nil, err
	}
	if len(eps.Subsets) == 0 {
		return nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", svc.Name))
	}

	seenEndpointsToAddresses := func(scheme, port string) (sets.String, error) {
		ret := sets.String{}
		for _, ep := range excludedEndpoints {
			if ep.Scheme == scheme && ep.Port() == port {
				ip, _, err := net.SplitHostPort(ep.Host)
				if err != nil {
					return nil, err
				}
				ret.Insert(ip)
			}
		}
		return ret, nil
	}

	allEndpoints := []*url.URL{}
	for ssi := 0; ssi < len(eps.Subsets); ssi++ {
		ss := &eps.Subsets[ssi]
		if len(ss.Addresses) == 0 {
			continue
		}
		for i := range ss.Ports {
			if ss.Ports[i].Name == svcPort.Name {
				port := int(ss.Ports[i].Port)

				seenAddresses, err := seenEndpointsToAddresses("https", strconv.Itoa(port))
				if err != nil {
					return nil, err
				}

				for _, ep := range ss.Addresses {
					if seenAddresses.Has(ep.IP) {
						continue
					}
					allEndpoints = append(allEndpoints, &url.URL{
						Scheme: "https",
						Host:   net.JoinHostPort(ep.IP, strconv.Itoa(port)),
					})
				}
			}
		}
	}

	if len(allEndpoints) == 0 {
		return nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
	}
	return allEndpoints, nil
}

func ResolveCluster(services listersv1.ServiceLister, namespace, id string, port int32) (*url.URL, error) {
	svc, err := services.Services(namespace).Get(id)
	if err != nil {
		return nil, err
	}

	switch {
	case svc.Spec.Type == v1.ServiceTypeClusterIP && svc.Spec.ClusterIP == v1.ClusterIPNone:
		return nil, fmt.Errorf(`cannot route to service with ClusterIP "None"`)
	// use IP from a clusterIP for these service types
	case svc.Spec.Type == v1.ServiceTypeClusterIP, svc.Spec.Type == v1.ServiceTypeLoadBalancer, svc.Spec.Type == v1.ServiceTypeNodePort:
		svcPort, err := findServicePort(svc, port)
		if err != nil {
			return nil, err
		}
		return &url.URL{
			Scheme: "https",
			Host:   net.JoinHostPort(svc.Spec.ClusterIP, fmt.Sprintf("%d", svcPort.Port)),
		}, nil
	case svc.Spec.Type == v1.ServiceTypeExternalName:
		return &url.URL{
			Scheme: "https",
			Host:   net.JoinHostPort(svc.Spec.ExternalName, fmt.Sprintf("%d", port)),
		}, nil
	default:
		return nil, fmt.Errorf("unsupported service type %q", svc.Spec.Type)
	}
}
