/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package lb

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"strings"

	compute "google.golang.org/api/compute/v1"
	gce "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

type gceUrlMap map[string]map[string]*compute.BackendService

type L7s struct {
	*ClusterManager
	pool *poolStore
}

func NewL7Pool(c *ClusterManager) *L7s {
	return &L7s{c, newPoolStore()}
}

func (l *L7s) create(name string) (*L7, error) {
	return &L7{
		Name:           name,
		cloud:          l.cloud,
		defaultBackend: l.defaultBackend,
	}, nil
}

func lbName(key string) string {
	return strings.Replace(key, "/", "", -1)
}

func (l *L7s) Get(name string) (*L7, error) {
	name = lbName(name)
	lb, exists := l.pool.Get(name)
	if !exists {
		return nil, fmt.Errorf("Loadbalancer %v not in pool", name)
	}
	return lb.(*L7), nil
}

func (l *L7s) Add(name string) (err error) {
	name = lbName(name)
	lb, _ := l.Get(name)
	if lb == nil {
		glog.Infof("Creating l7 %v", name)
		lb, err = l.create(name)
		if err != nil {
			return err
		}
	}
	// Why edge hop for the create?
	// The loadbalancer is a fictitious resource, it doesn't exist in gce. To
	// make it exist we need to create a collection of gce resources, done
	// through the edge hop.
	if err := lb.edgeHop(); err != nil {
		return err
	}

	l.pool.Add(name, lb)
	return nil
}

func (l *L7s) Delete(name string) error {
	name = lbName(name)
	lb, err := l.Get(name)
	if err != nil {
		return err
	}
	glog.Infof("Deleting lb %v", lb.Name)
	if err := lb.Cleanup(); err != nil {
		return err
	}
	l.pool.Delete(lb.Name)
	return nil
}

// Sync loadbalancers with the given names.
func (l *L7s) Sync(names []string) error {
	glog.Infof("Syincing loadbalancers %+v", names)

	knownLoadBalancers := util.NewStringSet()
	for _, n := range names {
		knownLoadBalancers.Insert(lbName(n))
	}
	pool := l.pool.snapshot()

	// Delete unknown loadbalancers
	for name, _ := range pool {
		if knownLoadBalancers.Has(name) {
			continue
		}
		if err := l.Delete(name); err != nil {
			return err
		}
	}

	// create new loadbalancers, perform an edge hop for existing
	for _, n := range names {
		if err := l.Add(n); err != nil {
			return err
		}
	}
	return nil
}

// L7 represents a single L7 loadbalancer.
type L7 struct {
	Name  string
	cloud *gce.GCECloud
	um    *compute.UrlMap
	tp    *compute.TargetHttpProxy
	fw    *compute.ForwardingRule
	// This is the backend to use if no path rules match
	// TODO: Expose this to users.
	defaultBackend *compute.BackendService
}

func (l *L7) checkUrlMap(backend *compute.BackendService) (err error) {
	if l.defaultBackend == nil {
		return fmt.Errorf("Cannot create urlmap without default backend.")
	}
	urlMapName := fmt.Sprintf("%v-%v", urlMapPrefix, l.Name)
	urlMap, _ := l.cloud.GetUrlMap(urlMapName)
	if urlMap != nil {
		glog.Infof("Url map %v already exists", urlMap.Name)
		l.um = urlMap
		return nil
	}

	glog.Infof("Creating url map for backend %v", l.defaultBackend.Name)
	urlMap, err = l.cloud.CreateUrlMap(l.defaultBackend, urlMapName)
	if err != nil {
		return err
	}
	l.um = urlMap
	return nil
}

func (l *L7) checkProxy() (err error) {
	if l.um == nil {
		return fmt.Errorf("Cannot create proxy without urlmap.")
	}
	proxyName := fmt.Sprintf("%v-%v", targetProxyPrefix, l.Name)
	proxy, _ := l.cloud.GetProxy(proxyName)
	if proxy == nil {
		glog.Infof("Creating new http proxy for urlmap %v", l.um.Name)
		proxy, err = l.cloud.CreateProxy(l.um, proxyName)
		if err != nil {
			return err
		}
		l.tp = proxy
		return nil
	}
	if !compareLinks(proxy.UrlMap, l.um.SelfLink) {
		glog.Infof("Proxy %v has the wrong url map, setting %v overwriting %v",
			proxy.Name, l.um, proxy.UrlMap)
		if err := l.cloud.SetUrlMapForProxy(proxy, l.um); err != nil {
			return err
		}
	}
	l.tp = proxy
	return nil
}

func (l *L7) checkForwardingRule() (err error) {
	if l.tp == nil {
		return fmt.Errorf("Cannot create forwarding rule without proxy.")
	}

	forwardingRuleName := fmt.Sprintf("%v-%v", forwardingRulePrefix, l.Name)
	fw, _ := l.cloud.GetGlobalForwardingRule(forwardingRuleName)
	if fw == nil {
		glog.Infof("Creating forwarding rule for proxy %v", l.tp.Name)
		fw, err = l.cloud.CreateGlobalForwardingRule(
			l.tp, forwardingRuleName, defaultPortRange)
		if err != nil {
			return err
		}
		l.fw = fw
		return nil
	}
	// TODO: If the port range and protocol don't match, recreate the rule
	if compareLinks(fw.Target, l.tp.SelfLink) {
		glog.Infof("Forwarding rule %v already exists", fw.Name)
		l.fw = fw
		return nil
	}
	glog.Infof("Forwarding rule %v has the wrong proxy, setting %v overwriting %v",
		fw.Name, fw.Target, l.tp.SelfLink)
	if err := l.cloud.SetProxyForGlobalForwardingRule(fw, l.tp); err != nil {
		return err
	}
	l.fw = fw
	return nil
}

func (l *L7) edgeHop() error {
	if err := l.checkUrlMap(l.defaultBackend); err != nil {
		return err
	}
	if err := l.checkProxy(); err != nil {
		return err
	}
	if err := l.checkForwardingRule(); err != nil {
		return err
	}
	return nil
}

// GetIP returns the ip associated with the forwarding rule for this l7.
func (l *L7) GetIP() string {
	return l.fw.IPAddress
}

// getNameForPathMatcher returns a name for a pathMatcher based on the given host rule.
// The host rule can be a regex, the path matcher name used to associate the 2 cannot.
func getNameForPathMatcher(hostRule string) string {
	hasher := md5.New()
	hasher.Write([]byte(hostRule))
	return fmt.Sprintf("%v%v", hostRulePrefix, hex.EncodeToString(hasher.Sum(nil)))
}

// UpdateUrlMap translates the given hostname: endpoint->port mapping into a gce url map.
//
// The GCE url map allows multiple hosts to share url->backend mappings without duplication, eg:
//   Host: foo(PathMatcher1), bar(PathMatcher1,2)
//   PathMatcher1:
//     /a -> b1
//     /b -> b2
//   PathMatcher2:
//     /c -> b1
// This leads to a lot of complexity in the common case, where all we want is a mapping of
// host->{/path: backend}.
//
// Consider some alternatives:
// 1. Using a single backend per PathMatcher:
//   Host: foo(PathMatcher1,3) bar(PathMatcher1,2,3)
//   PathMatcher1:
//     /a -> b1
//   PathMatcher2:
//     /c -> b1
//   PathMatcher3:
//     /b -> b2
// 2. Using a single host per PathMatcher:
//   Host: foo(PathMatcher1)
//   PathMatcher1:
//     /a -> b1
//     /b -> b2
//   Host: bar(PathMatcher2)
//   PathMatcher2:
//     /a -> b1
//     /b -> b2
//     /c -> b1
// In the context of kubernetes services, 2 makes more sense, because we
// rarely want to lookup backends (service:nodeport). When a service is
// deleted, we need to find all host PathMatchers that have the backend
// and remove the mapping. When a new path is added to a host (happens
// more frequently than service deletion) we just need to lookup the 1
// pathmatcher of the host.
func (l *L7) UpdateUrlMap(subdomainToBackendUrlMap gceUrlMap) error {
	if l.um == nil {
		return fmt.Errorf("Cannot add url without an urlmap.")
	}
	glog.Infof("Updating urlmap for l7 %v", l.Name)

	for hostname, urlToBackend := range subdomainToBackendUrlMap {
		// Find the hostrule
		// Find the path matcher
		// Add all given endpoint:backends to pathRules in path matcher
		var hostRule *compute.HostRule
		pmName := getNameForPathMatcher(hostname)
		for _, hr := range l.um.HostRules {
			// TODO: Hostnames must be exact match?
			if hr.Hosts[0] == hostname {
				hostRule = hr
				break
			}
		}
		if hostRule == nil {
			// This is a new host
			hostRule = &compute.HostRule{
				Hosts:       []string{hostname},
				PathMatcher: pmName,
			}
			l.um.HostRules = append(l.um.HostRules, hostRule)
		}
		var pathMatcher *compute.PathMatcher
		for _, pm := range l.um.PathMatchers {
			if pm.Name == hostRule.PathMatcher {
				pathMatcher = pm
				break
			}
		}
		if pathMatcher == nil {
			// This is a dangling or new host
			pathMatcher = &compute.PathMatcher{
				Name:           pmName,
				DefaultService: l.um.DefaultService,
			}
			l.um.PathMatchers = append(l.um.PathMatchers, pathMatcher)
		}
		// Clobber existing path rules.
		pathMatcher.PathRules = []*compute.PathRule{}
		for ep, bg := range urlToBackend {
			pathMatcher.PathRules = append(pathMatcher.PathRules, &compute.PathRule{[]string{ep}, bg.SelfLink})
		}
	}
	if um, err := l.cloud.UpdateUrlMap(l.um); err != nil {
		return err
	} else {
		l.um = um
	}
	return nil
}

// Cleanup deletes resources specific to this l7 in the right order.
// forwarding rule -> target proxy -> url map
// This leaves backends and health checks, which are shared across loadbalancers.
func (l *L7) Cleanup() error {
	if l.fw != nil {
		glog.Infof("Deleting global forwarding rule %v", l.fw.Name)
		if err := l.cloud.DeleteGlobalForwardingRule(l.fw.Name); err != nil {
			return err
		}
		l.fw = nil
	}
	if l.tp != nil {
		glog.Infof("Deleting target proxy %v", l.tp.Name)
		if err := l.cloud.DeleteProxy(l.tp.Name); err != nil {
			return err
		}
		l.tp = nil
	}
	if l.um != nil {
		glog.Infof("Deleting url map %v", l.um.Name)
		if err := l.cloud.DeleteUrlMap(l.um.Name); err != nil {
			return err
		}
		l.um = nil
	}
	return nil
}
