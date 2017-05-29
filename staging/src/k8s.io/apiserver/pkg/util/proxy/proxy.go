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

	"k8s.io/apimachinery/pkg/api/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/pkg/api/v1"
)

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func ResolveEndpoint(services listersv1.ServiceLister, endpoints listersv1.EndpointsLister, namespace, id string) (*url.URL, error) {
	// Allow ID as "svcname", "svcname:port", or "scheme:svcname:port".
	svcScheme, svcName, portStr, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	// If a port *number* was specified, find the corresponding service port name
	if portNum, err := strconv.ParseInt(portStr, 10, 64); err == nil {
		svc, err := services.Services(namespace).Get(svcName)
		if err != nil {
			return nil, err
		}
		found := false
		for _, svcPort := range svc.Spec.Ports {
			if int64(svcPort.Port) == portNum {
				// use the declared port's name
				portStr = svcPort.Name
				found = true
				break
			}
		}
		if !found {
			return nil, errors.NewServiceUnavailable(fmt.Sprintf("no service port %d found for service %q", portNum, svcName))
		}
	}

	eps, err := endpoints.Endpoints(namespace).Get(svcName)
	if err != nil {
		return nil, err
	}
	if len(eps.Subsets) == 0 {
		return nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", svcName))
	}
	// Pick a random Subset to start searching from.
	ssSeed := rand.Intn(len(eps.Subsets))
	// Find a Subset that has the port.
	for ssi := 0; ssi < len(eps.Subsets); ssi++ {
		ss := &eps.Subsets[(ssSeed+ssi)%len(eps.Subsets)]
		if len(ss.Addresses) == 0 {
			continue
		}
		for i := range ss.Ports {
			if ss.Ports[i].Name == portStr {
				// Pick a random address.
				ip := ss.Addresses[rand.Intn(len(ss.Addresses))].IP
				port := int(ss.Ports[i].Port)
				return &url.URL{
					Scheme: svcScheme,
					Host:   net.JoinHostPort(ip, strconv.Itoa(port)),
				}, nil
			}
		}
	}
	return nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
}

func ResolveCluster(services listersv1.ServiceLister, namespace, id string) (*url.URL, error) {
	if len(id) == 0 {
		return &url.URL{Scheme: "https"}, nil
	}

	destinationHost := id + "." + namespace + ".svc"
	service, err := services.Services(namespace).Get(id)
	if err != nil {
		return &url.URL{
			Scheme: "https",
			Host:   destinationHost,
		}, nil
	}
	switch {
	// use IP from a clusterIP for these service types
	case service.Spec.Type == v1.ServiceTypeClusterIP,
		service.Spec.Type == v1.ServiceTypeNodePort,
		service.Spec.Type == v1.ServiceTypeLoadBalancer:
		return &url.URL{
			Scheme: "https",
			Host:   service.Spec.ClusterIP,
		}, nil
	}
	return &url.URL{
		Scheme: "https",
		Host:   destinationHost,
	}, nil
}
