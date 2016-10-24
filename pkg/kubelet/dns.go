/*
Copyright 2015 The Kubernetes Authors.

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

// TODO: Move this into network when the import cycle is eliminated:
package kubelet

import (
	"fmt"
	"net"
	"os"

	"k8s.io/kubernetes/pkg/api"
)

// DNSPlugin acts as an interface between kubelet and any DNS resolvers.
type DNSPlugin interface {
	// getIP returns the ip the local/remote placed in resolv.conf
	GetResolvers() ([]string, error)
	GetSearchPaths(p *api.Pod) ([]string, error)
}

type hostResolver struct {
	// When the kubelet --resolv-conf flag is set to the empty string, use
	// DNS settings that override the docker default (which is to use
	// /etc/resolv.conf) and effectively disable DNS lookups. According to
	// the bind documentation, the behavior of the DNS client library when
	// "nameservers" are not specified is to "use the nameserver on the
	// local machine". A nameserver setting of localhost is equivalent to
	// this documented behavior.
	fallback *rawResolver
	kl       *Kubelet
}

func (h *hostResolver) GetResolvers() ([]string, error) {
	if h.kl.resolverConfig == "" {
		return h.fallback.GetResolvers()
	}
	f, err := os.Open(h.kl.resolverConfig)
	if err != nil {
		r, _ := h.fallback.GetResolvers()
		return r, err
	}
	defer f.Close()

	var hostDNS []string
	hostDNS, _, err = h.kl.parseResolvConf(f)
	if err != nil {
		r, _ := h.fallback.GetResolvers()
		return r, err
	}
	return hostDNS, nil
}

func (h *hostResolver) GetSearchPaths(pod *api.Pod) ([]string, error) {
	if h.kl.resolverConfig == "" {
		return h.fallback.GetSearchPaths(pod)
	}
	f, err := os.Open(h.kl.resolverConfig)
	if err != nil {
		s, _ := h.fallback.GetSearchPaths(pod)
		return s, err
	}
	defer f.Close()

	var hostPaths []string
	_, hostPaths, err = h.kl.parseResolvConf(f)
	if err != nil {
		s, _ := h.fallback.GetSearchPaths(pod)
		return s, err
	}
	return hostPaths, nil
}

// bridgeBoundResolver just returns the first IP in the bridge.
// This is the container bridge. It assumes a resolver is actually
// listening on this IP.
type bridgeBoundResolver struct {
	bridgeName    string
	clusterDomain string
}

func (p *bridgeBoundResolver) GetResolvers() ([]string, error) {
	iface, err := net.InterfaceByName(p.bridgeName)
	if err != nil {
		return []string{}, fmt.Errorf("Failed to retrieve bridge %v: %v", p.bridgeName, err)
	}
	addrs, err := iface.Addrs()
	if err != nil {
		return []string{}, fmt.Errorf("Failed to parse bridge %v addresses %v: %v", p.bridgeName, addrs, err)
	}
	if len(addrs) == 0 {
		return []string{}, fmt.Errorf("Bridge %v has no address to bind to", p.bridgeName)
	}
	for _, address := range addrs {
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			// Take the first ip in the CIDR.
			ip4 := ipnet.IP.To4()
			if ip4 != nil {
				// The bridge is assigned the first ip in the CIDR.
				// podCIDR starts from 0 though, so it would need an increment.
				// ip4[3]++
				return []string{ip4.String()}, nil
			}
		}
	}
	return []string{}, fmt.Errorf("No bridge, cannot determine %v ip for local resolver.", p.bridgeName)
}

func (p *bridgeBoundResolver) GetSearchPaths(pod *api.Pod) ([]string, error) {
	if p.clusterDomain != "" {
		nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, p.clusterDomain)
		svcDomain := fmt.Sprintf("svc.%s", p.clusterDomain)
		return append([]string{nsSvcDomain, svcDomain, p.clusterDomain}), nil
	}
	return []string{}, fmt.Errorf("No cluster domain, cannot find search paths")
}

// clusterFirst returns the cluster DNS resolver and search paths.
type clusterFirst struct {
	fallback      *rawResolver
	clusterDomain string
}

func (c *clusterFirst) GetResolvers() ([]string, error) {
	return c.fallback.GetResolvers()
}

func (c *clusterFirst) GetSearchPaths(pod *api.Pod) ([]string, error) {
	if c.clusterDomain != "" {
		nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, c.clusterDomain)
		svcDomain := fmt.Sprintf("svc.%s", c.clusterDomain)
		return append([]string{nsSvcDomain, svcDomain, c.clusterDomain}), nil
	}
	return []string{}, fmt.Errorf("No cluster domain, cannot find search paths")
}

// rawResolver just returns the IP it is seeded with.
// Usually this IP doesn't change and is initialized at kubelet boot time.
type rawResolver struct {
	ip net.IP
}

func (n *rawResolver) GetResolvers() ([]string, error) {
	if n.ip == nil {
		return []string{}, fmt.Errorf("No ip, cannot determine local resolver bind ip.")
	}
	return []string{n.ip.String()}, nil
}

func (p *rawResolver) GetSearchPaths(pod *api.Pod) ([]string, error) {
	return []string{"."}, nil
}
